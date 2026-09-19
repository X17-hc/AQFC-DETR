"""Manual first-step AdamW diagnostic: update disposable parameter copies only."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_geometry_gradients import Tee, gradient_summary, cosine


def select_images(annotation, count=32):
    ids = sorted(x['id'] for x in annotation['images'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs')
    counts = dict.fromkeys(ids, 0)
    for a in annotation['annotations']:
        if not a.get('ignore', 0) and not a.get('iscrowd', 0):
            counts[a['image_id']] += 1
    pools = [[] for _ in range(5)]
    for i in ids:
        n = counts[i]
        pools[0 if n == 0 else 1 if n <= 100 else 2 if n <= 300 else 3 if n <= 900 else 4].append(i)
    rng = random.Random(42)
    for pool in pools:
        rng.shuffle(pool)
    original = [len(p) for p in pools]
    selected = []
    while len(selected) < count:
        before = len(selected)
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(pool.pop())
        if before == len(selected):
            raise ValueError('Not enough images')
    return selected, dict(population=original, selected=[original[i]-len(p) for i, p in enumerate(pools)],
                          gt_counts={i: counts[i] for i in selected}, method='seed42 round-robin density strata; diagnostic oversampling')


def module_name(name):
    if 'bbox_embed' in name:
        return 'encoder_bbox' if 'enc_out_bbox' in name else 'decoder_bbox'
    for key, label in [('backbone', 'backbone'), ('query_allocator', 'allocator'),
                       ('feature_calibrator', 'calibrator'), ('density_pyramid', 'calibrator'),
                       ('transformer.decoder', 'decoder_other'), ('transformer.encoder', 'encoder_other')]:
        if key in name:
            return label
    return 'other'


def disposable_step(named, groups, loss, amp, scale, clip, retain_graph):
    """Use real AdamW/GradScaler on clones; original parameters never receive .grad."""
    import torch
    clones = {id(p): torch.nn.Parameter(p.detach().clone()) for _, p in named}
    copied_groups = [{**g, 'params': [clones[id(p)] for p in g['params']]} for g in groups]
    optimizer = torch.optim.AdamW(copied_groups)
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=scale)
    gradients = torch.autograd.grad(scaler.scale(loss), [p for _, p in named],
                                    retain_graph=retain_graph, allow_unused=True)
    for (_, p), g in zip(named, gradients):
        clones[id(p)].grad = None if g is None else g.detach()
    del gradients
    if amp:
        scaler.unscale_(optimizer)
    norm = gradient_summary([clones[id(p)].grad for _, p in named])
    torch.nn.utils.clip_grad_norm_(list(clones.values()), clip if clip > 0 else float('inf'), error_if_nonfinite=True)
    old_scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    if scaler.get_scale() < old_scale:
        raise FloatingPointError('AMP skipped virtual update; stop without retry')
    delta = [(clones[id(p)].detach() - p.detach()).float().cpu() for _, p in named]
    gradient_summary(delta)
    metadata = dict(gradient_norm=norm, clip_factor=min(1., clip/(norm+1e-6)) if clip > 0 else 1.,
                    scale_before=old_scale, scale_after=scaler.get_scale(), update_applied=True,
                    optimizer_initial_state='fresh_empty', updated_parameters=len(optimizer.state))
    return delta, metadata


def compare_updates(named, a, b):
    labels = ['all'] + sorted(set(module_name(n) for n, _ in named))
    result = {}
    for label in labels:
        ix = [i for i, (n, _) in enumerate(named) if label == 'all' or module_name(n) == label]
        x, y = [a[i] for i in ix], [b[i] for i in ix]
        na, nb = gradient_summary(x), gradient_summary(y)
        difference = gradient_summary([v-u for u, v in zip(x, y)])
        result[label] = dict(c0_update_norm=na, c1_update_norm=nb, difference_norm=difference,
                             relative_difference=None if na == 0 else difference/na,
                             cosine=cosine(x, y), tensors=len(ix),
                             changed_elements=sum(int((u != v).sum()) for u, v in zip(x, y)),
                             elements=sum(v.numel() for v in x))
    return result


def run(cli, output, record):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from main import build_model_main
    from datasets import build_dataset
    from datasets.coco import make_coco_transforms
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    from util.incremental_checkpoint import validate_warmstart
    from util.experiment import sha256
    from util.get_param_dicts import get_param_dict
    from util.misc import collate_fn
    from engine import AllocatorWeightScheduler
    from models.aqfcdetr.query_allocator import QueryBudgetLoss

    config = SLConfig.fromfile(cli.config)._cfg_dict.to_dict()
    validate_config(config)
    config.update(coco_path=cli.data_root, device='cuda', distributed=False, rank=0,
        fix_size=False, masks=False, pretrain_model_path=cli.pretrained, num_workers=0,
        data_aug_scales=[800], data_aug_max_size=1333, mosaic_p=0., copy_paste_p=0.)
    args = argparse.Namespace(**config)
    if args.geometry_loss_weight != .05 or args.training_phase_fixed_epoch != 23:
        raise ValueError('Requires frozen geometry_v1 C1 configuration')
    annotation = Path(cli.data_root)/'annotations/aitodv2_trainval.json'
    hashes = {str(p): sha256(p) for p in (Path(cli.pretrained), annotation)}
    if hashes[cli.pretrained] != args.expected_pretrained_sha256:
        raise ValueError('Wrong common epoch23 checkpoint')
    ids, sampling = select_images(json.loads(annotation.read_text()), cli.images)
    record.update(inputs=hashes, config=config, image_ids=ids, sampling=sampling,
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), torch=torch.__version__, cuda=torch.version.cuda,
        scope='First AdamW step, fresh optimizer per arm and per batch. Not mature momentum or training.',
        limitations='Plateau geometry weight .05, validation transforms, no AMP equivalence claim; no loss-group attribution in this entry.')
    if cli.check_only:
        record['status'] = 'preflight_only'
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fallback')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, criterion, _ = build_model_main(args)
    validate_warmstart(model, args)
    saved = torch.load(cli.pretrained, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True); del saved
    model.cuda().train(); model.set_epoch(23); criterion.cuda().train()
    criterion.quality_lambda = .25; criterion.weight_dict['loss_geometry'] = .05
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    versions = {n: p._version for n, p in named}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    groups = get_param_dict(args, model)
    for group in groups:
        group.setdefault('lr', args.lr); group.setdefault('weight_decay', args.weight_decay)
    grouped = [id(p) for g in groups for p in g['params']]
    if len(grouped) != len(set(grouped)) or set(grouped) != {id(p) for _, p in named}:
        raise ValueError('Optimizer groups do not partition trainable parameters')
    record['optimizer_groups'] = [{**{k:v for k,v in g.items() if k != 'params'},
        'names':[n for n,p in named if any(p is q for q in g['params'])]} for g in groups]
    budget = QueryBudgetLoss(coverage_weight=args.coverage_loss_weight,
        spacing_weight=args.spacing_loss_weight, count_weight=args.count_loss_weight,
        interval_weight=args.interval_loss_weight, boundary_guide_weight=args.boundary_guide_loss_weight,
        density_weight=args.density_map_loss_weight, density_target_backend=args.density_target_backend,
        density_target_chunk_size=args.density_target_chunk_size).cuda()
    allocator_weight = AllocatorWeightScheduler(**args.allocator_schedule, total_epochs=24).get_weight(23,0,1)*args.allocator_loss_weight
    dataset = build_dataset('trainval', args)
    dataset._transforms = make_coco_transforms('val', fix_size=False, strong_aug=False, args=args)
    dataset.filter_empty_gt = False
    index = {v:i for i,v in enumerate(dataset.ids)}
    loader = DataLoader(Subset(dataset,[index[i] for i in ids]), batch_size=2,
                        num_workers=0, collate_fn=collate_fn)
    record.update(gpu=str(torch.cuda.get_device_properties(0)), allocator_weight=allocator_weight,
                  checkpoint_writes=0, persistent_optimizer_updates=0, virtual_optimizer_updates=0)
    with (output/'updates.jsonl').open('x',encoding='utf-8') as stream:
        for batch,(samples,targets) in enumerate(loader):
            samples=samples.to('cuda')
            targets=[{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
            for mode in cli.precisions.split(','):
                with torch.no_grad():
                    for n,b in model.named_buffers(): b.copy_(buffers[n])
                random.seed(42+batch); np.random.seed(42+batch)
                torch.manual_seed(42+batch); torch.cuda.manual_seed_all(42+batch)
                with torch.amp.autocast('cuda',enabled=mode=='amp'):
                    out=model(samples,targets); losses=criterion(out,targets)
                    allocation=budget(out['allocator_outputs'],dict(targets=targets,
                        real_counts=torch.tensor([len(t['labels']) for t in targets],device='cuda')))
                    base=sum(v*criterion.weight_dict[k] for k,v in losses.items()
                        if k in criterion.weight_dict and k!='loss_geometry')+allocator_weight*allocation['loss_allocator_total']
                    combined=base+.05*losses['loss_geometry']
                if not torch.isfinite(combined): raise FloatingPointError('Non-finite loss')
                # Both arms reuse this exact graph/matching. Only cloned parameters are stepped.
                a, ma=disposable_step(named,groups,base,mode=='amp',args.amp_init_scale,args.clip_max_norm,True)
                b, mb=disposable_step(named,groups,combined,mode=='amp',args.amp_init_scale,args.clip_max_norm,False)
                record['virtual_optimizer_updates']+=2
                row=dict(batch=batch,precision=mode,image_ids=[int(t['image_id']) for t in targets],
                    gt_count=sum(len(t['labels']) for t in targets),query_counts=out['executed_query_counts'].tolist(),
                    c0=ma,c1=mb,modules=compare_updates(named,a,b))
                stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
                print(f'[update] {batch+1}/{len(loader)} {mode}: relative parameter-update difference={row["modules"]["all"]["relative_difference"]}',flush=True)
                del a,b,base,combined,losses,out,allocation
    with torch.no_grad():
        for n,b in model.named_buffers(): b.copy_(buffers[n])
    if any(p._version!=versions[n] or p.grad is not None for n,p in named):
        raise AssertionError('Source model changed')
    if any(sha256(p)!=h for p,h in hashes.items()): raise AssertionError('Input changed')
    record.update(status='completed',source_model_unchanged=True,inputs_unchanged=True,buffers_restored=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data-root','pretrained','output-dir'): p.add_argument('--'+key,required=True)
    p.add_argument('--images',type=int,default=32)
    p.add_argument('--precisions',choices=['fp32','amp','fp32,amp'],default='fp32,amp')
    p.add_argument('--check-only',action='store_true')
    cli=p.parse_args()
    if cli.images<2 or cli.images>32 or cli.images%2: p.error('images must be even, 2..32')
    from datetime import datetime, timezone
    import uuid
    import hashlib
    output=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    record=dict(status='running',command=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    begin=time.perf_counter()
    with (output/'console.log').open('x',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print(f'Output: {output}\nDisposable first-step updates only; no checkpoint writes.',flush=True)
            try: run(cli,output,record)
            except Exception as exc:
                import traceback
                record.update(status='failed',error=repr(exc)); traceback.print_exc(); raise
            finally:
                record['elapsed_seconds']=time.perf_counter()-begin
                (output/'manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            print(f'Status: {record["status"]}; elapsed {record["elapsed_seconds"]:.2f}s')


if __name__=='__main__': main()
