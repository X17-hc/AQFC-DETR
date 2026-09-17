"""Evaluation-only observer for the approved epoch23 subset diagnosis.

Deploy as tools/legacy24_diagnosis.py. No training implementation is provided.
The observer never changes model outputs and uses GT only after forward returns.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

EXPECTED='c447367e2d19411319a990985e0127af8db08bfb7eec0329b2065b095ccc64eb'
ANNOTATION='9817a75f9bc4a84015881f2ddbf39bcc29635654a0266b210fd382743c927d98'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def bucket(n): return 0 if n==0 else 1 if n<=100 else 2 if n<=300 else 3 if n<=900 else 4

def sample(annotation,total=1000):
    counts=Counter(a['image_id'] for a in annotation['annotations'] if not a.get('ignore',0) and not a.get('iscrowd',0))
    groups=[[] for _ in range(5)]
    ids=[i['id'] for i in annotation['images']]
    if len(ids)!=len(set(ids)) or len(ids)<total: raise ValueError('Invalid population')
    for i in sorted(ids): groups[bucket(counts[i])].append(i)
    quota=[min(100,len(g)) for g in groups]
    if sum(quota)>total: raise ValueError('Total below base quotas')
    remain=total-sum(quota); capacities=[len(g)-q for g,q in zip(groups,quota)]
    shares=[remain*c/sum(capacities) if remain else 0 for c in capacities]
    extra=[math.floor(s) for s in shares]
    for i in sorted(range(5),key=lambda i:(-(shares[i]-extra[i]),i))[:remain-sum(extra)]: extra[i]+=1
    quota=[q+e for q,e in zip(quota,extra)]
    rng=random.Random(42)
    result=sorted(i for g,q in zip(groups,quota) for i in rng.sample(g,q))
    return result,dict(population=list(map(len,groups)),sample=quota,seed=42,gt_definition='exclude ignore/crowd')

def proposal_record(outputs,image,ground_truth):
    import numpy as np
    from util.error_analysis import box_iou_xywh
    valid=outputs['query_valid_mask'][0].detach().cpu().numpy().astype(bool)
    k=int(outputs['executed_query_counts'][0])
    if int(valid.sum())!=k: raise ValueError('Invalid query mask')
    boxes=outputs['interm_outputs']['pred_boxes'][0].detach().float().cpu().numpy()[valid].copy()
    if not np.isfinite(boxes).all(): raise ValueError('Nonfinite proposal')
    boxes[:,:2]-=boxes[:,2:]/2
    boxes*=np.array([image['width'],image['height'],image['width'],image['height']])
    if (boxes[:,2:]<=0).any(): raise ValueError('Nonpositive proposal size')
    best=np.zeros(len(ground_truth),dtype=float)
    for start in range(0,len(boxes),256):
        if ground_truth: best=np.maximum(best,box_iou_xywh(boxes[start:start+256],[g['bbox'] for g in ground_truth]).max(0))
    return dict(image_id=image['id'],executed_query_count=k,proposal_stage='selected_encoder_regressed_boxes_before_decoder',
        coordinate_format='original_pixel_xywh',proposals=boxes.tolist(),
        ground_truth=[dict(annotation_id=g['id'],category_id=g['category_id'],area=g['area'],bbox=g['bbox'],best_proposal_iou=float(v)) for g,v in zip(ground_truth,best)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--image-ids',type=Path)
    p.add_argument('--mode',choices=['prepare','evaluate'],required=True)
    p.add_argument('--group',choices=['D0','D1','D2'],default='D0')
    p.add_argument('--export-proposals',action='store_true')
    p.add_argument('--capture-raw',action='store_true',help='Only for the fixed32 precheck')
    p.add_argument('--fp32',action='store_true')
    a=p.parse_args(); root=a.project_root.resolve(); sys.path.insert(0,str(root))
    annotation=a.data_root/'annotations/aitodv2_test.json'
    if digest(a.checkpoint)!=EXPECTED: raise ValueError('Unexpected checkpoint SHA256')
    if digest(annotation)!=ANNOTATION: raise ValueError('Unexpected test annotation SHA256')
    ann=read(annotation)
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    if a.mode=='prepare':
        ids,strata=sample(ann)
        a.output_dir.mkdir(parents=True)
        save(a.output_dir/'subset_ids.json',ids)
        save(a.output_dir/'precheck_ids.json',sorted(random.Random(42).sample(ids,32)))
        save(a.output_dir/'sampling.json',strata)
        print(strata); return
    if not a.image_ids: raise ValueError('Subset is mandatory; full test is forbidden')
    ids=read(a.image_ids); images={x['id']:x for x in ann['images']}
    if len(ids) not in (32,1000) or len(set(ids))!=len(ids) or not set(ids)<=set(images): raise ValueError('Invalid subset')
    if a.capture_raw and len(ids)!=32: raise ValueError('Raw capture is precheck only')
    gt=defaultdict(list)
    seen=set()
    for g in ann['annotations']:
        if g['image_id'] in ids:
            if g.get('ignore',0) or g.get('iscrowd',0): raise ValueError('Selected ignore/crowd needs separate matching protocol')
            if g['id'] in seen: raise ValueError('Duplicate GT ID')
            seen.add(g['id']); gt[g['image_id']].append(g)
    import torch
    import main as training_entry
    a.output_dir.mkdir(parents=True)
    source={str(f.relative_to(root)):digest(f) for folder in ['models','util','datasets','configs','tools'] for f in (root/folder).rglob('*.py')}
    source.update({name:digest(root/name) for name in ['main.py','engine.py']})
    provenance=dict(command=sys.argv,checkpoint_sha256=EXPECTED,annotation_sha256=ANNOTATION,subset_sha256=digest(a.image_ids),
        source_sha256=source,group=a.group,started_at=datetime.now(timezone.utc).isoformat(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),amp=not a.fp32,purpose='test engineering diagnosis',
        prediction_scope='native postprocess top-k; no added score threshold; not complete logits',
        config_sha256=digest(a.config))
    provenance['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used','--format=csv'],text=True)
    save(a.output_dir/'diagnosis_manifest.json',provenance)
    command=['--config',str(a.config),'--data-root',str(a.data_root),'--resume',str(a.checkpoint),'--no-pretrained','--eval',
        '--output-dir',str(a.output_dir),'--device','cuda','--seed','42','--num_workers','0','--eval-image-ids',str(a.image_ids),
        '--export-predictions','--export-diagnostics','--no-amp' if a.fp32 else '--amp','--options',
        'data_aug_scales=[800]','data_aug_max_size=1333']
    if a.group!='D0': command+=['force_query_budget=900']
    if a.group=='D2': command+=['proposal_selection_mode=semantic']
    args=argparse.ArgumentParser(parents=[training_entry.get_args_parser()]).parse_args(command)
    assert args.eval and not args.eval_ema
    original=training_entry.evaluate
    def observed(model,criterion,postprocessors,data_loader,*rest,**kwargs):
        if list(data_loader.dataset.ids)!=ids or data_loader.batch_size!=1: raise ValueError('Unexpected loader/order')
        count=0
        with ExitStack() as stack:
            stream=stack.enter_context((a.output_dir/'proposals.jsonl').open('x',encoding='utf-8')) if a.export_proposals else None
            if a.capture_raw: (a.output_dir/'raw').mkdir()
            def hook(module,inputs,outputs):
                nonlocal count
                if count>=len(ids): raise ValueError('Extra model forward')
                image_id=ids[count]
                if stream:
                    record=proposal_record(outputs,images[image_id],gt[image_id])
                    stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n'); stream.flush()
                if a.capture_raw:
                    torch.save({k:outputs[k].detach().cpu().clone() for k in ['pred_logits','pred_boxes','query_valid_mask']},a.output_dir/'raw'/f'{image_id}.pth')
                count+=1
                if count%50==0 or count==len(ids): print(f'[diagnosis] {count}/{len(ids)}',flush=True)
            handle=model.register_forward_hook(hook)
            try: result=original(model,criterion,postprocessors,data_loader,*rest,**kwargs)
            finally: handle.remove()
        if count!=len(ids): raise ValueError('Incomplete forwards')
        return result
    training_entry.evaluate=observed
    start=time.time()
    try:
        training_entry.main(args)
        if digest(a.checkpoint)!=EXPECTED or digest(annotation)!=ANNOTATION: raise ValueError('Input changed')
        for name,value in source.items():
            if digest(root/name)!=value: raise ValueError('Source changed during evaluation: '+name)
        save(a.output_dir/'diagnosis_status.json',dict(status='completed',images=len(ids),elapsed_seconds=time.time()-start))
    except BaseException as exc:
        save(a.output_dir/'diagnosis_status.json',dict(status='failed',error=repr(exc),elapsed_seconds=time.time()-start))
        raise

if __name__=='__main__': main()
