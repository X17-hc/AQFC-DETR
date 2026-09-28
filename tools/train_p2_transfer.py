"""MANUAL: two-epoch frozen-source adapter training. Does not invoke any detector decoder."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import build_model_main
from datasets import build_dataset
from engine import MosaicPScheduler
from util.slconfig import SLConfig
from util.config_validation import validate_config
from util.incremental_checkpoint import validate_warmstart
from util.precision24 import NEW_PREFIXES
from util.p2_transfer import VERSION,SOURCE_SHA,freeze_for_adaptation,frozen_digest,extract_inputs
from util.checkpoint import capture_rng_state,restore_rng_state
from util.experiment import sha256
from util.misc import collate_fn
from models.aqfcdetr.p2_transfer import encoder_forward,adaptation_loss


def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


def validate_recipe(values):
    validate_config(values)
    if values.get('p2_transfer_version')!=VERSION or values.get('p2_transfer_mode')!='reference':
        raise ValueError('Adaptation requires layerwise_v2 reference config')
    if values['epochs']!=2 or values['train_split']!='trainval' or values['training_phase_fixed_epoch']!=23:
        raise ValueError('Adaptation recipe requires two epochs, trainval and fixed phase23')
    if values['batch_size']!=2 or values['lr']!=1e-4 or values['train_transform_mode']!='native_multiscale':
        raise ValueError('Adaptation recipe differs from the approved recipe')
    # native_multiscale deliberately ignores inherited legacy data_aug_scales.
    # Validate the executed resize, rather than mistakenly rejecting the S2 recipe.
    from datasets.native_resize import NativeMultiScaleResize
    resize=NativeMultiScaleResize()
    if (list(resize.sizes)!=[640,704,768,800] or resize.max_size!=1333 or
            values['weight_decay']!=1e-4 or values['clip_max_norm']!=.1):
        raise ValueError('Adaptation scales/optimizer must remain fixed')


def build_frozen(cli):
    if sha256(cli.pretrained)!=SOURCE_SHA:
        raise ValueError('Adaptation source must be the frozen S2 SHA-256')
    values=SLConfig.fromfile(cli.config)._cfg_dict.to_dict()
    validate_recipe(values)
    values.update(coco_path=cli.data_root,device=cli.device,distributed=False,rank=0,
                  fix_size=False,masks=False,pretrain_model_path=cli.pretrained)
    args=argparse.Namespace(**values)
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,_,_=build_model_main(args)
    validate_warmstart(model,args)
    saved=torch.load(cli.pretrained,map_location='cpu',weights_only=False)['model']
    result=model.load_state_dict(saved,strict=False)
    if result.unexpected_keys or any(not k.startswith(NEW_PREFIXES) for k in result.missing_keys):
        raise ValueError('Unexpected warm-start state')
    model.to(cli.device)
    freeze_for_adaptation(model)
    model.p2_initialization_report=dict(source_checkpoint=cli.pretrained,source_sha256=SOURCE_SHA,
        loaded_state_keys=len(saved),new_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,
        loaded_parameter_numel=sum(p.numel() for n,p in model.named_parameters() if n in saved),
        total_parameter_numel=sum(p.numel() for p in model.parameters()),
        trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad])
    return args,model


def restore_adapter_training(path,model,optimizer,scaler,signature):
    data=torch.load(path,map_location='cpu',weights_only=False)
    if (data.get('kind')!='p2_adaptation_resume' or not data.get('epoch_complete') or
            data.get('signature')!=signature or data.get('epoch') not in (0,1)):
        raise ValueError('Only matching complete adaptation epochs may resume')
    if data['frozen_digest']!=frozen_digest(model): raise ValueError('Frozen source state differs')
    model.transformer.encoder.p2_substitutes.load_state_dict(data['adapters'],strict=True)
    optimizer.load_state_dict(data['optimizer']); scaler.load_state_dict(data['scaler'])
    restore_rng_state(data['rng'])
    model.p2_previous_metrics=data.get('completed_metrics',[])
    return data['epoch']+1,data['successful_updates']


def save_checkpoint(path,data):
    temporary=path.with_suffix('.tmp')
    torch.save(data,temporary)
    os.replace(temporary,path)


def train(cli,out):
    if cli.max_train_steps<0 or cli.max_train_steps==1 or cli.workers<0:
        raise ValueError('Use 0 for full adaptation, or at least two total smoke steps')
    if cli.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    seed_all()
    args,model=build_frozen(cli)
    dataset=build_dataset(args.train_split,args)
    schedule=MosaicPScheduler(peak_p=args.mosaic_p,warmup_end=args.mosaic_warmup_end,
        decay_start=args.mosaic_decay_start,total_epochs=24)
    if dataset.mosaic is not None: dataset.mosaic.p=schedule.get_p(23)
    print(f'Fixed phase23: Mosaic={dataset.mosaic.p if dataset.mosaic else 0}, CopyPaste={dataset.copy_paste_p}',flush=True)
    loader=DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=cli.workers,
        collate_fn=collate_fn,pin_memory=True,persistent_workers=False)
    if not len(loader): raise ValueError('Empty adaptation loader')
    adapters=model.transformer.encoder.p2_substitutes
    optimizer=torch.optim.AdamW(adapters.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=cli.amp,init_scale=32.)
    baseline=frozen_digest(model)
    signature=dict(version=VERSION,source_sha256=SOURCE_SHA,config=values_for_signature(args),
        annotation_sha256=sha256(Path(cli.data_root)/'annotations/aitodv2_trainval.json'),
        dataset_root=str(Path(cli.data_root).resolve()),amp=cli.amp,workers=cli.workers,
        batch=2,seed=42,loss='rms_smooth_l1_1_cosine_025_gaussian_1plus4_v1')
    start,success=0,0
    if cli.resume:
        if cli.max_train_steps: raise ValueError('Smoke and resume cannot be combined')
        start,success=restore_adapter_training(cli.resume,model,optimizer,scaler,signature)
        if start>=2: raise ValueError('Two-epoch adaptation already complete')
    manifest=dict(command=sys.argv,signature=signature,resolved_config=vars(args),frozen_digest=baseline,
        initialization=model.p2_initialization_report,
        actual_augmentation=dict(mosaic=dataset.mosaic.p if dataset.mosaic else 0.,copy_paste=dataset.copy_paste_p),
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__),ROOT/'models/aqfcdetr/p2_transfer.py',ROOT/'util/p2_transfer.py']},
        smoke=bool(cli.max_train_steps),results=[],
        resumed_from=dict(path=cli.resume,sha256=sha256(cli.resume)) if cli.resume else None,
        previous_completed_metrics=getattr(model,'p2_previous_metrics',[]))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    for epoch in range(start,2):
        phase='teacher_forced' if epoch==0 else 'free_running'
        freeze_for_adaptation(model)
        epoch_start=time.perf_counter(); skips=0; applied=0; streak=0; loss_sum=0.; steps=0
        timing=dict(features=0.,reference=0.,student_forward=0.,backward_optimizer=0.)
        smoke_steps = cli.max_train_steps//2 if epoch==0 else cli.max_train_steps-cli.max_train_steps//2
        limit=min(smoke_steps,len(loader)) if cli.max_train_steps else len(loader)
        iterator=iter(loader)
        for step in range(limit):
            samples,targets=next(iterator)
            samples=samples.to(cli.device)
            targets=[{k:v.to(cli.device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
            optimizer.param_groups[0]['lr']=1e-4*min(1.,.1+.9*success/500)
            optimizer.zero_grad(set_to_none=True)
            events=[torch.cuda.Event(enable_timing=True) for _ in range(5)] if cli.device.startswith('cuda') else None
            cpu_times=[time.perf_counter()]
            if events: events[0].record()
            with torch.autocast(device_type=cli.device.split(':')[0],enabled=cli.amp):
                inputs=extract_inputs(model,samples)
                if events: events[1].record()
                cpu_times.append(time.perf_counter())
                with torch.no_grad():
                    _,reference=encoder_forward(model.transformer.encoder,*inputs,mode='reference',trace=True)
                if events: events[2].record()
                cpu_times.append(time.perf_counter())
                loss,parts=adaptation_loss(model.transformer.encoder,inputs,targets,phase,reference)
                if events: events[3].record()
                cpu_times.append(time.perf_counter())
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite transfer loss')
            before=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(adapters.parameters(),.1)
            scaler.step(optimizer); scaler.update()
            ok=scaler.get_scale()>=before
            if not cli.amp and not torch.isfinite(grad): raise FloatingPointError('Non-finite gradient')
            success+=int(ok); applied+=int(ok); skips+=int(not ok); streak=0 if ok else streak+1
            if streak>=20: raise FloatingPointError('20 consecutive AMP skips; no retry')
            if events:
                events[4].record(); events[4].synchronize()
                durations=[events[i].elapsed_time(events[i+1])/1000 for i in range(4)]
            else:
                cpu_times.append(time.perf_counter()); durations=np.diff(cpu_times).tolist()
            for key,elapsed in zip(timing,durations): timing[key]+=elapsed
            loss_sum+=float(loss.detach()); steps+=1
            if step%1000==0 or step+1==limit:
                print(json.dumps(dict(epoch=epoch,phase=phase,step=step+1,total=limit,
                    loss=float(loss.detach()),layer_losses=parts.float().tolist(),lr=optimizer.param_groups[0]['lr'],
                    successful_updates=success,amp_skips=skips,grad_norm_before_clip=float(grad))),flush=True)
        del iterator
        if not applied: raise RuntimeError('No successful optimizer updates')
        if frozen_digest(model)!=baseline: raise RuntimeError('Frozen model parameters or buffers changed')
        complete=not bool(cli.max_train_steps) and steps==len(loader)
        result=dict(epoch=epoch,phase=phase,steps=steps,full_epoch_steps=len(loader),epoch_complete=complete,
            successful_updates=applied,amp_skips=skips,mean_loss=loss_sum/steps,
            training_seconds=time.perf_counter()-epoch_start,cuda_segment_seconds=timing,
            timing_note='adaptation CUDA events; excludes data wait, includes one end-step sync; not detector throughput')
        checkpoint=dict(kind='p2_adaptation_resume',version=VERSION,signature=signature,
            epoch=epoch,epoch_complete=complete,adapters={k:v.detach().cpu() for k,v in adapters.state_dict().items()},
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=capture_rng_state(),
            successful_updates=success,frozen_digest=baseline,phase=phase,
            completed_metrics=manifest['previous_completed_metrics']+manifest['results']+[result])
        save_checkpoint(out/f'checkpoint{epoch:04d}.pth',checkpoint)
        manifest['results'].append(result)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
        print(f'[Adaptation completed] {json.dumps(result)}',flush=True)
    if not cli.max_train_steps:
        save_checkpoint(out/'adapter_initialization.pth',dict(kind='p2_adapter_initialization',version=VERSION,
            source_sha256=SOURCE_SHA,complete_two_epochs=True,signature=signature,
            adapters={k:v.detach().cpu() for k,v in adapters.state_dict().items()},
            successful_updates=success,frozen_digest=baseline,acceptance='NOT_EVALUATED',
            completed_metrics=manifest['previous_completed_metrics']+manifest['results']))
    print('Done. No detector training or automatic evaluation started.',flush=True)


def values_for_signature(args):
    return {k:v for k,v in vars(args).items() if k not in ('device','distributed','rank','pretrain_model_path','coco_path')}


class Tee:
    def __init__(self,stream,file): self.stream,self.file=stream,file
    def write(self,text): self.stream.write(text); self.file.write(text); self.file.flush()
    def flush(self): self.stream.flush(); self.file.flush()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/p2_transfer/adapt_2e.py'))
    p.add_argument('--data-root',required=True); p.add_argument('--pretrained',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--resume',default='')
    p.add_argument('--max-train-steps',type=int,default=0)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--device',default='cuda')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    return p


def main():
    cli=parser().parse_args()
    out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    with (out/'console.log').open('x',encoding='utf-8') as log:
        stdout,stderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print(f'Output: {out}',flush=True)
            train(cli,out)
        except BaseException:
            traceback.print_exc(); raise
        finally:
            sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__': main()
