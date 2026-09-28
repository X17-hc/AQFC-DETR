"""Complete-epoch stop and explicit F2 first-epoch continuation artifacts."""
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid
import xml.etree.ElementTree as ET


def validate_stop(args):
    limit = getattr(args, 'stop_after_epochs', 0)
    if type(limit) is not int or limit < 0:
        raise ValueError('--stop-after-epochs must be a nonnegative integer')
    if limit and (args.eval or args.max_train_steps or args.max_eval_steps or args.debug or not args.output_dir):
        raise ValueError('--stop-after-epochs requires full training/evaluation, an output directory, and no debug/step limits')


def reached_stop(args, epoch):
    limit = getattr(args, 'stop_after_epochs', 0)
    return bool(limit and epoch - args.start_epoch + 1 >= limit)


def atomic_save(checkpoint, path):
    """Same-directory atomic replacement; preserve previous checkpoint on failure."""
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Only our exact temporary file, never a user checkpoint.
        if temporary.exists():
            temporary.unlink()


def first_epoch_review(args, train, evaluation, full_steps, full_images, image_ids, checkpoint, root):
    from .experiment import sha256
    stats = evaluation or {}
    values = stats.get('coco_eval_bbox', [])
    ap, vt = (100*values[0], 100*values[4]) if len(values) > 4 else (None, None)
    valid = ap is not None and vt is not None and all(math.isfinite(x) and x >= 0 for x in (ap, vt))
    ids = list(image_ids)
    checks = dict(full_training=train.get('train_iterations') == full_steps == 7009,
        optimizer_updated=train.get('optimizer_steps', 0) > 0,
        update_accounting=train.get('optimizer_steps', 0) + train.get('amp_skipped_steps', 0) == train.get('train_iterations'),
        full_evaluation=stats.get('evaluated_images') == full_images == len(set(ids)) == len(ids) == 14018,
        no_step_limits=not (args.max_train_steps or args.max_eval_steps or args.debug),
        metrics_finite=valid, AP_no_large_drop=valid and ap >= 31.878707245513593-1,
        APvt_no_large_drop=valid and vt >= 15.696368085310372-1,
        checkpoint_exists=Path(checkpoint).is_file())
    passed = all(checks.values())
    result = dict(status='READY_FOR_MANUAL_REVIEW' if passed else 'PAUSE_AND_REVIEW', checks=checks,
        AP=ap if valid else None, APvt=vt if valid else None, checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint) if Path(checkpoint).is_file() else None,
        note='Engineering screening only; no automatic continuation or causal accuracy claim.')
    out = Path(args.output_dir)
    if passed:
        # Explicit exact checkpoint; never select a latest directory or silently warm-start.
        argv = ['--config', str(Path(args.config_file).resolve()), '--data-root', str(args.coco_path),
                '--output-dir', str(out.resolve()), '--resume', str(Path(checkpoint).resolve()),
                '--device', args.device, '--amp' if args.amp else '--no-amp', '--seed', str(args.seed),
                '--num_workers', str(args.num_workers), '--max-train-steps', '0', '--max-eval-steps', '0',
                '--stop-after-epochs', '0', '--max-consecutive-skipped-steps', str(args.max_consecutive_skipped_steps)]
        command = [sys.executable, '-u', str(root/'tools/run_f2_training.py'), *argv]
        result['resume_command'] = command
        (out/'resume_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        (out/'resume_command.txt').write_text(shlex.join(command)+'\n', encoding='utf-8')
        component = ET.Element('component', name='ProjectRunConfigurationManager')
        cfg = ET.SubElement(component, 'configuration', default='false',
            name='AQFC-DETR F2六层细化续训至24轮及最终评估', type='PythonConfigurationType', factoryName='Python')
        ET.SubElement(cfg, 'module', name='AQFC-DETR')
        envs = ET.SubElement(cfg, 'envs')
        for k, v in {'PYTHONUNBUFFERED':'1', 'PYTHONIOENCODING':'utf-8',
                     'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','2'),
                     'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS','2'),
                     'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS','2')}.items():
            ET.SubElement(envs, 'env', name=k, value=v)
        options = dict(INTERPRETER_OPTIONS='-u', PARENT_ENVS='true',
            SDK_HOME='/opt/conda/envs/AQFC-DETR/bin/python',
            SDK_NAME='SSH (sftp://root@219.216.64.62:32880/opt/conda/envs/AQFC-DETR/bin/python)',
            WORKING_DIRECTORY=str(root), IS_MODULE_SDK='false', ADD_CONTENT_ROOTS='true', ADD_SOURCE_ROOTS='true',
            SCRIPT_NAME='$PROJECT_DIR$/tools/run_f2_training.py', PARAMETERS=shlex.join(argv),
            MODULE_MODE='false', EMULATE_TERMINAL='false')
        for k,v in options.items(): ET.SubElement(cfg,'option',name=k,value=v)
        ET.SubElement(cfg,'method',v='2')
        ET.indent(component)
        ET.ElementTree(component).write(out/'F2_resume_24e.run.xml', encoding='utf-8', xml_declaration=True)
    (out/'first_epoch_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result
