"""MANUAL ONLY: matched real-image training-step benchmark, no checkpoint updates."""
import argparse
import contextlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import build_model_main
from datasets import build_dataset
from datasets.coco import make_coco_transforms
from util.slconfig import SLConfig
from util.config_validation import validate_config
from util.incremental_checkpoint import validate_warmstart
from util.misc import collate_fn, MetricLogger
from util.experiment import unique_output, sha256
from models.aqfcdetr.query_allocator import QueryBudgetLoss
from util.get_param_dicts import get_param_dict
from util.profiling import training_profile


def run(variant,workers,repeat,cli,ids):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    config=SLConfig.fromfile(str(ROOT/f'configs/incremental_v2/{variant}.py'))._cfg_dict.to_dict()
    validate_config(config)
    config.update(coco_path=cli.data_root,device='cuda',distributed=False,rank=0,
        fix_size=False,masks=False,pretrain_model_path=cli.pretrained,num_workers=workers,
        data_aug_scales=[800],data_aug_max_size=1333,mosaic_p=0.,copy_paste_p=0.)
    args=argparse.Namespace(**config)
    # Training members with validation transforms: deterministic geometry across workers.
    dataset=build_dataset('trainval',args)
    dataset._transforms=make_coco_transforms('val',fix_size=False,strong_aug=False,args=args)
    dataset.filter_empty_gt=False
    index={image_id:i for i,image_id in enumerate(dataset.ids)}
    loader=DataLoader(Subset(dataset,[index[i] for i in ids]),batch_size=2,
        num_workers=workers,collate_fn=collate_fn,pin_memory=True,persistent_workers=False)
    model,criterion,_=build_model_main(args)
    validate_warmstart(model,args)
    state=torch.load(cli.pretrained,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True); del state
    model.cuda().train(); model.set_epoch(11); criterion.cuda().train()
    criterion.quality_lambda=.25 if variant=='p2' else 0.
    optimizer=torch.optim.AdamW(get_param_dict(args,model),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=cli.amp,init_scale=32.)
    budget=QueryBudgetLoss(coverage_weight=args.coverage_loss_weight,spacing_weight=args.spacing_loss_weight,
        count_weight=args.count_loss_weight,interval_weight=args.interval_loss_weight,
        boundary_guide_weight=args.boundary_guide_loss_weight,density_weight=args.density_map_loss_weight,
        density_target_backend=args.density_target_backend,density_target_chunk_size=args.density_target_chunk_size).cuda()
    from engine import AllocatorWeightScheduler
    weight=AllocatorWeightScheduler(**args.allocator_schedule,total_epochs=24).get_weight(11,0,len(loader))*args.allocator_loss_weight
    timings=[]; core=[]; calls=[]; skips=0; streak=0; total_skips=0
    measured_start_unix=None
    meter=MetricLogger(batched_transfer=args.batched_metric_transfer)
    trace=str(Path(cli.output_dir)/f'{variant}_w{workers}_r{repeat}.json') if cli.profile else ''
    iterator=iter(loader)
    with training_profile(trace,model,criterion) if cli.profile else contextlib.nullcontext():
        for step in range(cli.warmup+cli.steps):
            torch.cuda.synchronize(); begin=time.perf_counter()
            if step==cli.warmup: measured_start_unix=time.time()
            samples,targets=next(iterator)
            if getattr(cli,'require_square_800',False) and list(samples.tensors.shape)!=[2,3,800,800]:
                raise ValueError('Controlled benchmark requires actual [2,3,800,800]; no silent resize fallback')
            core_begin=time.perf_counter()
            samples=samples.to('cuda',non_blocking=args.non_blocking_transfer)
            targets=[{k:v.to('cuda',non_blocking=args.non_blocking_transfer) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=cli.amp):
                out=model(samples,targets); losses=criterion(out,targets)
                loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
                alloc=budget(out['allocator_outputs'],dict(targets=targets,real_counts=torch.tensor([len(t['labels']) for t in targets],device='cuda')))
                loss=loss+weight*alloc['loss_allocator_total']
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite benchmark loss')
            scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_max_norm)
            scaler.step(optimizer); scaler.update()
            meter.update(**{f'{k}_unscaled':v for k,v in losses.items()},
                         **{k:v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict})
            applied=scaler.get_scale()>=scale
            total_skips+=int(not applied)
            streak=0 if applied else streak+1
            if streak>=20: raise FloatingPointError('20 consecutive benchmark optimizer skips')
            torch.cuda.synchronize(); end=time.perf_counter()
            if step==cli.warmup-1: torch.cuda.reset_peak_memory_stats()
            if step>=cli.warmup:
                timings.append(end-begin); core.append(end-core_begin); skips+=int(not applied)
                calls.append(dict(shape=list(samples.tensors.shape),queries=out['executed_query_counts'].tolist(),
                                  step_seconds=end-begin,core_seconds=end-core_begin,
                                  optimizer_applied=bool(applied),
                                  image_ids=ids[2*step:2*step+2]))
    result=dict(variant=variant,workers=workers,repeat=repeat,images_per_second=2/statistics.mean(timings),
        mean_step_seconds=statistics.mean(timings),mean_h2d_model_loss_backward_optimizer_seconds=statistics.mean(core),
        p50_step_seconds=float(np.percentile(timings,50)),p90_step_seconds=float(np.percentile(timings,90)),
        allocated_mb=torch.cuda.max_memory_allocated()/2**20,reserved_mb=torch.cuda.max_memory_reserved()/2**20,
        amp_skips=skips,successful_updates=cli.steps-skips,
        total_successful_updates=cli.warmup+cli.steps-total_skips,total_amp_skips=total_skips,
        measured_start_unix=measured_start_unix,measured_end_unix=time.time(),
        resolved_config=config,quality_lambda=criterion.quality_lambda,measurements=calls)
    del model,criterion,optimizer,budget,out,loss,losses,alloc,loader,iterator
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True); p.add_argument('--pretrained',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--steps',type=int,default=200); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--profile',action='store_true',help='Separate instrumented pass, not a speed result')
    args=p.parse_args()
    if min(args.warmup,args.steps,args.repeats)<1: p.error('Counts must be positive')
    if args.profile: args.warmup=1; args.steps=20; args.repeats=1
    args.unique_output_dir=True; args.resume=''; unique_output(args)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    annotation=Path(args.data_root)/'annotations/aitodv2_trainval.json'
    all_ids=sorted(x['id'] for x in json.loads(annotation.read_text())['images'])
    random.Random(42).shuffle(all_ids)
    ids=all_ids[:2*(args.warmup+args.steps)]
    if len(ids)<2*(args.warmup+args.steps): raise ValueError('Not enough images for requested benchmark')
    record=dict(purpose='profile' if args.profile else 'benchmark',results=[],image_ids=ids,
        checkpoint_sha256=sha256(args.pretrained),annotation_sha256=sha256(annotation),
        amp=args.amp,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),conditions='fixed trainval images; validation resize; no augmentation; transient optimizer updates only; core time includes H2D and scalar metrics; no checkpoint saved')
    target=Path(args.output_dir)/'benchmark.json'
    try:
        variants=[('repaired',0),('p1',0),('p1',2),('p1',4),('p1',8)] if not args.profile else [('p1',0)]
        for variant,workers in variants:
            for repeat in range(args.repeats):
                row=run(variant,workers,repeat,args,ids); record['results'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='measurements'}),flush=True)
                target.write_text(json.dumps(record,indent=2),encoding='utf-8')
        if not args.profile:
            speed={w:statistics.median(r['images_per_second'] for r in record['results'] if r['variant']=='p1' and r['workers']==w) for w in (0,2,4,8)}
            best=max(speed.values()); chosen=min(w for w,v in speed.items() if v>=best*.98)
            record['recommended_workers']=chosen
            for repeat in range(args.repeats): record['results'].append(run('p2',chosen,repeat,args,ids))
        record['status']='completed'
    except Exception as exc:
        record.update(status='failed',error=repr(exc)); raise
    finally:
        target.write_text(json.dumps(record,indent=2),encoding='utf-8')


if __name__=='__main__': main()
