"""Independent sample RNG, epoch-boundary loader recovery and manual continuation."""
from contextlib import contextmanager
import copy
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class SeededDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Includes all random calls inside Mosaic/Copy-Paste; no global reseeding
        # leaks into model dropout/DN or into a different worker's sample.
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{index}'.encode()).digest()[:4], 'little')
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def loader_state(loader):
    return dict(sampler=loader.batch_sampler.sampler.generator.get_state(), workers=loader.generator.get_state())


def restore_loader(loader, state):
    if state is not None:
        loader.batch_sampler.sampler.generator.set_state(state['sampler'])
        loader.generator.set_state(state['workers'])


def subset_evaluation(args, epoch, model, criterion, postprocessors, dataset, base_ds, device, evaluator):
    from util.misc import collate_fn
    path = Path(args.joint_subset_ids)
    raw = json.loads(path.read_text(encoding='utf-8'))
    ids = raw if isinstance(raw, list) else raw.get('image_ids', raw.get('ids'))
    if not isinstance(ids, list) or len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('H1 subset must contain exactly 1000 unique IDs')
    mapping = {int(v):i for i,v in enumerate(dataset.ids)}
    if set(ids)-set(mapping):
        raise ValueError('H1 subset contains unknown IDs')
    out = Path(args.output_dir)/f'subset_epoch{epoch:04}'
    out.mkdir(exist_ok=False)
    subargs = copy.copy(args)
    subargs.output_dir, subargs.max_eval_steps = str(out), 0
    loader = DataLoader(Subset(dataset, [mapping[i] for i in ids]), batch_size=1,
                        num_workers=0, collate_fn=collate_fn)
    stats, result = evaluator(model, criterion, postprocessors, loader, base_ds, device, str(out), args=subargs)
    if stats.get('evaluated_images') != 1000 or set(result.coco_eval['bbox'].params.imgIds) != set(ids):
        raise ValueError('H1 subset evaluation coverage differs')
    torch.save(result.coco_eval['bbox'].eval, out/'eval.pth')
    (out/'metrics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


def continuation(args, train, evaluation, checkpoint):
    from .experiment import sha256
    out = Path(args.output_dir)
    checks = dict(full_training=train.get('train_iterations') == 7009,
        updated=train.get('optimizer_steps', 0) > 0,
        accounting=train.get('optimizer_steps', 0)+train.get('amp_skipped_steps', 0)==train.get('train_iterations'),
        full_evaluation=evaluation.get('evaluated_images') == 14018, checkpoint=checkpoint.is_file())
    record = dict(checks=checks, status='MANUAL_REVIEW_REQUIRED',
        metrics=evaluation.get('coco_eval_bbox'), checkpoint_sha256=sha256(checkpoint),
        historical_reference=dict(AP=31.1323, APvt=15.2318),
        warning='Different training scales; not a controlled architecture effect. No automatic continuation.')
    if all(checks.values()):
        argv = ['--config', args.config_file, '--data-root', str(args.coco_path),
            '--resume', str(checkpoint.resolve()), '--output-dir', str(out.resolve()),
            '--device', args.device, '--amp' if args.amp else '--no-amp', '--seed', str(args.seed),
            '--num_workers', str(args.num_workers), '--stop-after-epochs','0',
            '--max-train-steps','0','--max-eval-steps','0','--max-consecutive-skipped-steps','20']
        command = [sys.executable, '-u', str(Path(__file__).resolve().parents[1]/'tools/run_joint_training.py'), *argv]
        (out/'resume_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        (out/'resume_command.txt').write_text(shlex.join(command)+'\n', encoding='utf-8')
        # Explicit artifact for manual import only; never added to an auto-run queue.
        import os
        import xml.etree.ElementTree as ET
        root=ET.Element('component',name='ProjectRunConfigurationManager')
        config=ET.SubElement(root,'configuration',default='false',name='AQFC-DETR H1续训epoch1至23',
                             type='PythonConfigurationType',factoryName='Python')
        ET.SubElement(config,'module',name='AQFC-DETR')
        envs=ET.SubElement(config,'envs')
        for key,value in dict(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','2'),
            PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2').items():
            ET.SubElement(envs,'env',name=key,value=value)
        options=dict(SDK_HOME='/opt/conda/envs/AQFC-DETR/bin/python',
            SDK_NAME='SSH (sftp://root@219.216.64.62:32880/opt/conda/envs/AQFC-DETR/bin/python)',
            WORKING_DIRECTORY='/workspace/AQFC-DETR',SCRIPT_NAME='$PROJECT_DIR$/tools/run_joint_training.py',
            PARAMETERS=shlex.join(argv),INTERPRETER_OPTIONS='-u',IS_MODULE_SDK='false',PARENT_ENVS='true',
            ADD_CONTENT_ROOTS='true',ADD_SOURCE_ROOTS='true',EMULATE_TERMINAL='false',MODULE_MODE='false')
        for key,value in options.items():ET.SubElement(config,'option',name=key,value=value)
        ET.SubElement(config,'method',v='2');ET.indent(root)
        ET.ElementTree(root).write(out/'H1_resume_epoch1_23.run.xml',encoding='utf-8',xml_declaration=True)
    (out/'epoch0_review.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
