"""Manual C0/S1 evaluation-only diagnosis. No training, retries or GPU binding."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_geometry_pair import read, save, digest, run_logged, eval_command, ANNOTATION_SHA, IDS_SHA, INITIAL_SHA
from legacy24_diagnosis import proposal_record

WEIGHTS = {'C0': '5d9d1a94d5a0828d59490725f3b98a5196d5bc577cf1e213da18d2c805f8e6de',
           'S1': 'b3c9dc260ff4213194bea30f7e5653ce5b25a1858fdcb3918bf97a42c9b460a1'}
CONFIGS = {'C0': 'configs/geometry_v1/c0_3e.py', 'S1': 'configs/scale_v1/s1_3e.py'}
RARE = {0, 1, 4, 7}


def select_images(annotation, base):
    population = {x['id'] for x in annotation['images']}
    if not base or any(type(i) is not int for i in base) or len(base) != len(set(base)) or not set(base) <= population:
        raise ValueError('Invalid base image IDs')
    anns = annotation['annotations']
    if len({g['id'] for g in anns}) != len(anns):
        raise ValueError('Duplicate annotation IDs')
    rare = [g for g in anns if g['category_id'] in RARE and 0 <= g['area'] <= 64
            and not g.get('ignore', 0) and not g.get('iscrowd', 0)]
    rare_ids = {g['image_id'] for g in rare}
    added = sorted(rare_ids - set(base))
    ids = sorted(set(base) | rare_ids)
    if not set(ids) <= population:
        raise ValueError('Unknown annotation image ID')
    id_set = set(ids)
    selected = [g for g in anns if g['image_id'] in id_set]
    if any(g.get('ignore', 0) or g.get('iscrowd', 0) for g in selected):
        raise ValueError('Selected ignore/crowd requires a separate diagnostic protocol')
    return ids, dict(base_ids=sorted(base), added_ids=added, rare_image_ids=sorted(rare_ids),
                    rare_gt_ids=sorted(g['id'] for g in rare), gt=len(selected), images=len(ids),
                    scope='base and GT-selected enrichment; never full-test AP', vt_area='0 <= area <= 64')


def validate_pair(configs, manifests, after_mode='native800'):
    differences = {k for k in configs[0].keys() | configs[1].keys() if configs[0].get(k) != configs[1].get(k)}
    if differences - {'config_file', 'output_dir', 'train_transform_mode'}:
        raise ValueError(f'Unexpected training differences: {differences}')
    for label, cfg, manifest in zip(('C0', 'S1'), configs, manifests):
        if cfg.get('train_transform_mode', 'legacy') != ('legacy' if label == 'C0' else after_mode):
            raise ValueError('Training transform mode mismatch')
        if cfg.get('geometry_loss_weight') != 0 or cfg.get('quality_blend_max') != .25:
            raise ValueError('Unexpected loss recipe')
        if cfg.get('force_query_budget') is not None or cfg.get('proposal_selection_mode') != 'spatial':
            raise ValueError('Keep original dynamic spatial queries')
        if cfg.get('eval_split') != 'test' or manifest['initialization']['sha256'] != INITIAL_SHA:
            raise ValueError('Unexpected split or initialization')
    if manifests[0]['annotations'] != manifests[1]['annotations']:
        raise ValueError('Training annotation sources differ')


def evaluate_stage(args):
    """Post-forward observer returns None: cannot replace predictions or use GT for routing."""
    import main as entry
    ids = read(args.output / 'image_ids.json')
    annotation = read(args.data_root / 'annotations/aitodv2_test.json')
    images = {x['id']: x for x in annotation['images']}
    gt = defaultdict(list)
    for g in annotation['annotations']:
        gt[g['image_id']].append(g)
    run = args.c0_run if args.stage == 'C0' else args.s1_run
    checkpoint = run / 'checkpoint0002.pth'
    weights = getattr(args, 'stage_weights', WEIGHTS)
    configs = getattr(args, 'stage_configs', CONFIGS)
    if digest(checkpoint) != weights[args.stage]:
        raise ValueError('Checkpoint SHA mismatch')
    dest = args.output / args.stage
    command = eval_command(ROOT, ROOT / configs[args.stage], checkpoint, args.data_root,
                           args.output / 'image_ids.json', dest)[3:]
    parsed = entry.get_args_parser().parse_args(command)
    assert parsed.eval and not parsed.eval_ema and parsed.no_pretrained
    original = entry.evaluate
    def observed(model, criterion, postprocessors, loader, *rest, **kwargs):
        if list(loader.dataset.ids) != ids or loader.batch_size != 1:
            raise ValueError('Unexpected evaluation order/batch')
        count = 0
        with (dest / 'proposals.jsonl').open('x', encoding='utf-8') as stream:
            def hook(module, inputs, outputs):
                nonlocal count
                if count >= len(ids): raise ValueError('Extra forward')
                image_id = ids[count]
                row = proposal_record(outputs, images[image_id], gt[image_id])
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()
                count += 1
                if count % 100 == 0 or count == len(ids):
                    print(f'[{args.stage}] {count}/{len(ids)}', flush=True)
            handle = model.register_forward_hook(hook)
            try: result = original(model, criterion, postprocessors, loader, *rest, **kwargs)
            finally: handle.remove()
        if count != len(ids): raise ValueError('Incomplete proposal export')
        return result
    entry.evaluate = observed
    try: entry.main(parsed)
    finally: entry.evaluate = original


def targeted_summary(output, sampling, after_label='S1', c0_root=None):
    from util.paired_head_errors import new_group, add_pair
    proposal = {}
    for label in ('C0', after_label):
        folder = Path(c0_root) if label == 'C0' and c0_root is not None else output / label
        rows = [json.loads(x) for x in (folder / 'proposals.jsonl').read_text().splitlines()]
        assert len(rows) == sampling['images'] and len({r['image_id'] for r in rows}) == len(rows)
        assert {r['image_id'] for r in rows} == set(sampling['base_ids']) | set(sampling['added_ids'])
        proposal[label] = {(r['image_id'],g['annotation_id']): g['best_proposal_iou'] for r in rows for g in r['ground_truth']}
    groups = defaultdict(new_group)
    counts = defaultdict(lambda: {'gt': 0, 'C0_hits025': 0, 'C0_hits050': 0, after_label+'_hits025': 0, after_label+'_hits050': 0})
    base, rare = set(sampling['base_ids']), set(sampling['rare_gt_ids'])
    seen = set()
    with (output / 'targeted_gt.jsonl').open('x', encoding='utf-8') as stream:
        for line in (output / 'paired' / 'paired_gt.jsonl').open(encoding='utf-8'):
            row = json.loads(line); key = row['image_id'], row['annotation_id']
            if key in seen: raise ValueError('Duplicate paired GT')
            seen.add(key)
            cohort = 'base1000' if row['image_id'] in base else 'added_images'
            buckets = ['union', cohort, cohort + '/class:' + str(row['category_id'])]
            if 0 <= row['area'] <= 64:
                buckets += ['official_vt/class:' + str(row['category_id'])]
            if row['annotation_id'] in rare: buckets += ['rare_vt_complete']
            row['C0_proposal_best_iou'] = proposal['C0'][key]
            row[after_label+'_proposal_best_iou'] = proposal[after_label][key]
            row['cohort'] = cohort
            for bucket in buckets:
                add_pair(groups[bucket], row['epoch0'], row['epoch1'])
                counts[bucket]['gt'] += 1
                for label in ('C0', after_label):
                    counts[bucket][label+'_hits025'] += int(proposal[label][key] >= .25)
                    counts[bucket][label+'_hits050'] += int(proposal[label][key] >= .50)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    if len(seen) != sampling['gt'] or any(set(p) != seen for p in proposal.values()):
        raise ValueError('GT/proposal coverage mismatch')
    save(output / 'targeted_summary.json', dict(labels={'epoch0':'C0','epoch1':after_label},
         sampling=sampling, groups=groups, proposal_coverage=counts,
         warning='Union is enriched, not full-test AP. Official VT includes area64; generic paired size buckets are half-open.'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('c0-run','s1-run','data-root','image-ids','output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--check-only', action='store_true', help='CPU preflight only; no inference or output directory')
    p.add_argument('--stage', choices=['C0','S1'], help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args.stage:
        if args.check_only: raise ValueError('check-only cannot evaluate')
        return evaluate_stage(args)
    annotation = args.data_root / 'annotations/aitodv2_test.json'
    if digest(annotation) != ANNOTATION_SHA or digest(args.image_ids) != IDS_SHA:
        raise ValueError('Frozen annotation/base subset mismatch')
    base = read(args.image_ids)
    if len(base) != 1000: raise ValueError('Expected original 1000 images')
    ids, sampling = select_images(read(annotation), base)
    if len(ids)>1105 or len(sampling['rare_gt_ids'])!=105:
        raise ValueError('Rare VT support differs from approved diagnosis')
    runs = [args.c0_run, args.s1_run]
    configs = [read(p/'config_args_all.json') for p in runs]
    manifests = [read(p/'experiment_manifest.json') for p in runs]
    validate_pair(configs, manifests)
    from util.slconfig import SLConfig
    import torch
    hashes = {str(p): digest(p) for p in [annotation,args.image_ids]}
    for label, run, saved_cfg in zip(('C0','S1'),runs,configs):
        current = json.loads(json.dumps(SLConfig.fromfile(str(ROOT/CONFIGS[label]))._cfg_dict.to_dict()))
        if any(saved_cfg.get(k)!=v for k,v in current.items()):
            raise ValueError(f'{label} config changed since training')
        checkpoint = run/'checkpoint0002.pth'
        if digest(checkpoint)!=WEIGHTS[label]: raise ValueError(f'{label} checkpoint SHA mismatch')
        ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ckpt.get('epoch')!=2 or not ckpt.get('run_metadata',{}).get('epoch_complete') or ckpt.get('run_metadata',{}).get('smoke_test'):
            raise ValueError('Incomplete checkpoint')
        del ckpt
        for p in [checkpoint,run/'config_args_all.json',run/'experiment_manifest.json']:
            hashes[str(p)] = digest(p)
    for p in [ROOT/'main.py',ROOT/'engine.py'] + [p for folder in ('models','util','datasets','configs','tools') for p in (ROOT/folder).rglob('*.py')]:
        hashes[str(p)] = digest(p)
    print(f'Preflight OK: {len(ids)} images ({len(sampling["added_ids"])} added), {sampling["gt"]} GT; 105 rare VT GT.',flush=True)
    if args.check_only: return
    if len([v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v.strip()])!=1:
        raise ValueError('Select one GPU in the run configuration')
    output = args.output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    save(output/'image_ids.json',ids); save(output/'sampling.json',sampling)
    state=dict(status='running',inputs_sha256=hashes,labels={'epoch0':'C0','epoch1':'S1'},command=sys.argv,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),purpose='test engineering diagnosis, no training')
    try:
        state['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,utilization.gpu','--format=csv'],text=True)
        print('Output:',output,flush=True)
        for label in ('C0','S1'):
            state['stage']=label; save(output/'pair_manifest.json',state)
            run_logged([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:], '--stage',label,'--output',str(output)],
                       output/(label+'_console.log'),ROOT)
        state['stage']='paired_analysis'; save(output/'pair_manifest.json',state)
        run_logged([sys.executable,'-u',str(ROOT/'tools/analyze_paired_head_errors.py'),
            '--before',str(output/'C0'),'--after',str(output/'S1'),'--annotations',str(annotation),
            '--image-ids',str(output/'image_ids.json'),'--expected-images',str(len(ids)),
            '--expected-gt',str(sampling['gt']),'--output',str(output/'paired')],output/'paired_console.log',ROOT)
        targeted_summary(output,sampling)
        if any(digest(Path(p))!=h for p,h in hashes.items()): raise ValueError('Inputs/source changed during run')
        state.update(status='completed',inputs_unchanged=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc)); raise
    finally: save(output/'pair_manifest.json',state)
    print('Completed: targeted_summary.json and targeted_gt.jsonl; epoch0=C0, epoch1=S1.',flush=True)


if __name__=='__main__': main()
