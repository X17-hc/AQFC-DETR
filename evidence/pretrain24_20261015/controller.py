from pathlib import Path
import os,sys,json,time,subprocess,hashlib,traceback,math
A=Path('/workspace/AQFC-DETR/experiments/aqfc_pretrain24_20261015');C=A/'code';PY='/opt/conda/envs/AQFC-DETR/bin/python'
def save(name,v):
 p=A/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(command,out,stage):
 out.mkdir(exist_ok=False)
 with (out/'console.log').open('w') as log:
  p=subprocess.Popen(command,cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT)
  save('status.json',dict(status='running',stage=stage,child_pid=p.pid,command=command,output=str(out),time=time.time()))
  rc=p.wait()
 if rc:raise RuntimeError(f'{stage} exited {rc}; see {out}/console.log')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-52e42290-a934-c527-006e-bf6dc2a2d5bb',OMP_NUM_THREADS='2',PYTHONPATH='/workspace/AQFC-DETR/experiments/aqfc_speed_20261012/dependencies')
base=[PY,'-u',str(C/'main.py'),'--data-root','/workspace/DQDetr/data/path/AITODv2','--pretrained','/workspace/AQFC-DETR/weights/legacy/pretrain_model.pth','--device','cuda','--amp','--seed','42','--num_workers','2','--max-consecutive-skipped-steps','20']
try:
 assert os.statvfs(A).f_bfree*os.statvfs(A).f_frsize > 40*2**30
 smoke=C/'configs/pretrain_smoke.py';smoke.write_text("_base_ = ['mature_pretrain24.py']\njoint_subset_epochs = []\n")
 out=A/'smoke20'
 if not (out/'checkpoint.pth').exists():run(base+['--config',str(smoke),'--output-dir',str(out),'--max-train-steps','20'],out,'smoke20')
 report=json.loads((out/'checkpoint_migration_report.json').read_text());assert report['allowlist_verified'] and not report['shape_mismatched_keys']
 import torch
 ck=torch.load(out/'checkpoint.pth',map_location='cpu',weights_only=False)
 assert all(torch.isfinite(x).all() for x in ck['model'].values() if x.is_floating_point())
 print('PROGRESS',ck['criterion_progress'],flush=True)
 rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()];assert math.isfinite(rows[-1]['train_loss'])
 assert ck['run_metadata']['train_iterations']==20,ck['run_metadata']
 assert ck['criterion_progress']['successful_updates']>=19,ck['criterion_progress']
 save('preflight_result.json',dict(passed=True,criterion_progress=ck['criterion_progress'],metadata=ck['run_metadata'],loss=rows[-1]['train_loss'],coverage=report['parameter_only_coverage']))
 del ck;import gc;gc.collect()
 # Formal run starts fresh at seed42, not from the smoke checkpoint.
 out=A/'run_pretrain24';run(base+['--config',str(C/'configs/mature_pretrain24.py'),'--output-dir',str(out)],out,'formal24')
 rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()];assert rows[-1]['epoch']==23
 baseline=[json.loads(x) for x in (A/'baseline_metrics.jsonl').read_text().splitlines()];b=next(x for x in reversed(baseline) if 'test_coco_eval_bbox' in x);n=next(x for x in reversed(rows) if 'test_coco_eval_bbox' in x)
 save('comparison.json',dict(baseline=b['test_coco_eval_bbox'],pretrain=n['test_coco_eval_bbox'],delta_percentage_points=[100*(y-x) for x,y in zip(b['test_coco_eval_bbox'],n['test_coco_eval_bbox'])],baseline_run='/workspace/AQFC-DETR/outputs/legacy_joint/h1_aqfc_refactor_24e/20260930_134930_136792_0b5906dd508f'))
 save('status.json',dict(status='complete',last_epoch=23,time=time.time()))
except Exception:
 save('status.json',dict(status='failed',error=traceback.format_exc(),time=time.time()));raise

