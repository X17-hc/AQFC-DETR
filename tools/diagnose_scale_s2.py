"""Manual C0/S2 diagnosis: reuse frozen C0, evaluate S2 once, never train."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import diagnose_scale_pair as pair
from diagnose_geometry_pair import read,save,digest,run_logged,ANNOTATION_SHA,IDS_SHA
from analyze_paired_head_errors import resolve_export,validate_export

S2_SHA='40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928'
S2_CONFIG='configs/scale_v1/s2_3e.py'

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('c0-run','s2-run','reuse-pair','data-root','image-ids','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--stage',choices=['S2'],help=argparse.SUPPRESS)
    p.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    return p

def check_reuse(args,ids,sampling,annotation):
    old=read(args.reuse_pair/'pair_manifest.json')
    if old.get('status')!='completed' or not old.get('inputs_unchanged') or old.get('labels',{}).get('epoch0')!='C0':
        raise ValueError('C0 source diagnosis is not complete')
    if read(args.reuse_pair/'image_ids.json')!=ids or read(args.reuse_pair/'sampling.json')!=sampling:
        raise ValueError('Frozen sampling differs; do not rerun C0 automatically')
    old_hashes=old['inputs_sha256']
    if old_hashes.get(str(args.c0_run/'checkpoint0002.pth'))!=pair.WEIGHTS['C0']:
        raise ValueError('C0 provenance mismatch')
    # Require unchanged model/evaluation numerics. New train-only mode and
    # diagnostic wrapper are intentionally excluded, not silently overwritten.
    critical=[ROOT/'main.py',ROOT/'engine.py',ROOT/'datasets/transforms.py',ROOT/'util/box_ops.py']
    critical+=list((ROOT/'models').rglob('*.py'))
    for p in critical:
        if old_hashes.get(str(p))!=digest(p):
            raise ValueError(f'Historical inference source differs: {p}')
    folder=args.reuse_pair/'C0'; cfg=read(folder/'config_args_all.json')
    expected=dict(eval=True,eval_ema=False,amp=True,num_workers=0,max_eval_steps=0,
                  data_aug_scales=[800],data_aug_max_size=1333,force_query_budget=None,
                  proposal_selection_mode='spatial')
    for k,v in expected.items():
        if cfg.get(k)!=v: raise ValueError(f'C0 export protocol mismatch: {k}')
    if cfg.get('resume')!=str(args.c0_run/'checkpoint0002.pth'):
        raise ValueError('C0 export checkpoint mismatch')
    export=resolve_export(folder)
    validate_export(export,ids,{c['id']:c['name'] for c in annotation['categories']},
                    {x['id']:x for x in annotation['images']})
    seen=set(); actual_gt=set()
    for line in (folder/'proposals.jsonl').open():
        row=json.loads(line); image_id=row['image_id']
        if image_id in seen or image_id not in ids: raise ValueError('C0 proposal image mismatch')
        seen.add(image_id)
        for gt in row['ground_truth']:
            key=(image_id,gt['annotation_id'])
            if key in actual_gt: raise ValueError('Duplicate C0 proposal GT')
            actual_gt.add(key)
    required_gt={(g['image_id'],g['id']) for g in annotation['annotations'] if g['image_id'] in seen}
    if seen!=set(ids) or actual_gt!=required_gt: raise ValueError('C0 proposal GT coverage mismatch')
    return [args.reuse_pair/'pair_manifest.json',args.reuse_pair/'image_ids.json',args.reuse_pair/'sampling.json',
            folder/'config_args_all.json',folder/'proposals.jsonl',export/'metadata.json',export/'images.jsonl',export/'predictions.json']

def main():
    args=parser().parse_args()
    if args.stage:
        if args.check_only: raise ValueError('check-only cannot evaluate')
        args.s1_run=args.s2_run  # Existing evaluator's neutral second-run slot.
        args.stage_weights={'S2':S2_SHA}; args.stage_configs={'S2':S2_CONFIG}
        return pair.evaluate_stage(args)
    annotation_path=args.data_root/'annotations/aitodv2_test.json'
    if digest(annotation_path)!=ANNOTATION_SHA or digest(args.image_ids)!=IDS_SHA:
        raise ValueError('Frozen annotation/subset mismatch')
    annotation=read(annotation_path); base=read(args.image_ids)
    if len(base)!=1000: raise ValueError('Expected 1000 base images')
    ids,sampling=pair.select_images(annotation,base)
    if len(ids)!=1054 or sampling['gt']!=91192 or len(sampling['rare_gt_ids'])!=105:
        raise ValueError('Approved cohort mismatch')
    runs=[args.c0_run,args.s2_run]
    configs=[read(p/'config_args_all.json') for p in runs]
    manifests=[read(p/'experiment_manifest.json') for p in runs]
    pair.validate_pair(configs,manifests,after_mode='native_multiscale')
    from util.slconfig import SLConfig
    import torch
    inputs=[annotation_path,args.image_ids]
    for run,saved,path,sha in zip(runs,configs,[pair.CONFIGS['C0'],S2_CONFIG],[pair.WEIGHTS['C0'],S2_SHA]):
        current=json.loads(json.dumps(SLConfig.fromfile(str(ROOT/path))._cfg_dict.to_dict()))
        if any(saved.get(k)!=v for k,v in current.items()): raise ValueError('Training config changed')
        ck=run/'checkpoint0002.pth'
        if digest(ck)!=sha: raise ValueError('Checkpoint SHA mismatch')
        state=torch.load(ck,map_location='cpu',weights_only=False)
        if state.get('epoch')!=2 or not state.get('run_metadata',{}).get('epoch_complete') or state.get('run_metadata',{}).get('smoke_test'):
            raise ValueError('Incomplete checkpoint')
        del state
        inputs += [ck,run/'config_args_all.json',run/'experiment_manifest.json']
    inputs+=check_reuse(args,ids,sampling,annotation)
    inputs += [p for folder in ('models','util','datasets','configs','tools') for p in (ROOT/folder).rglob('*.py')]
    inputs += [ROOT/'main.py',ROOT/'engine.py']
    hashes={str(p):digest(p) for p in inputs}
    print('Preflight OK: reuse C0; S2 only; 1054 images / 91192 GT. Focus: VT vehicle/person.',flush=True)
    if args.check_only: return
    if len([x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()])!=1:
        raise ValueError('Select one GPU in run configuration')
    output=args.output_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    save(output/'image_ids.json',ids); save(output/'sampling.json',sampling)
    state=dict(status='running',labels={'epoch0':'C0','epoch1':'S2'},inputs_sha256=hashes,
      reused_c0=str(args.reuse_pair/'C0'),command=sys.argv,purpose='test engineering diagnosis; not full-test AP',
      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        state['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,utilization.gpu','--format=csv'],text=True)
        print('Output:',output,flush=True)
        state['stage']='S2'; save(output/'pair_manifest.json',state)
        run_logged([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--stage','S2','--output',str(output)],output/'S2_console.log',ROOT)
        state['stage']='paired_analysis'; save(output/'pair_manifest.json',state)
        run_logged([sys.executable,'-u',str(ROOT/'tools/analyze_paired_head_errors.py'),
          '--before',str(args.reuse_pair/'C0'),'--after',str(output/'S2'),'--annotations',str(annotation_path),
          '--image-ids',str(output/'image_ids.json'),'--expected-images','1054','--expected-gt','91192',
          '--output',str(output/'paired')],output/'paired_console.log',ROOT)
        pair.targeted_summary(output,sampling,after_label='S2',c0_root=args.reuse_pair/'C0')
        if any(digest(Path(p))!=h for p,h in hashes.items()): raise ValueError('Inputs changed during diagnosis')
        state.update(status='completed',inputs_unchanged=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc)); raise
    finally: save(output/'pair_manifest.json',state)
    print('Completed: epoch0=C0, epoch1=S2; targeted_summary.json includes all official VT classes.',flush=True)

if __name__=='__main__': main()
