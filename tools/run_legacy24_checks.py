"""Bounded stage launcher. No training, retries, full-test runs, or GPU selection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def compare(before,after,ids):
    maximum={'pred_logits':0.,'pred_boxes':0.,'scores':0.,'boxes':0.}
    for image_id in ids:
        x=torch.load(before/'raw'/f'{image_id}.pth',map_location='cpu',weights_only=True)
        y=torch.load(after/'raw'/f'{image_id}.pth',map_location='cpu',weights_only=True)
        assert torch.equal(x['query_valid_mask'],y['query_valid_mask'])
        for key in ['pred_logits','pred_boxes']:
            torch.testing.assert_close(x[key],y[key],atol=1e-5,rtol=1e-4)
            delta=(x[key]-y[key]).abs(); delta=delta[torch.isfinite(delta)]
            maximum[key]=max(maximum[key],float(delta.max()) if delta.numel() else 0.)
    files=[list(p.glob('predictions_*/predictions.json')) for p in [before,after]]
    assert all(len(f)==1 for f in files)
    x,y=[read(f[0]) for f in files]; assert len(x)==len(y)
    for a,b in zip(x,y):
        assert (a['image_id'],a['category_id'])==(b['image_id'],b['category_id'])
        for key,label in [('score','scores'),('bbox','boxes')]:
            np.testing.assert_allclose(a[key],b[key],atol=1e-5,rtol=1e-4)
            maximum[label]=max(maximum[label],float(np.max(np.abs(np.asarray(a[key])-b[key]))))
    return maximum

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['precheck','D0','D1','D2'],required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve(); out=a.output.resolve()
    checkpoint=root/'outputs/p2_legacy24/resume_epoch11_20260916/checkpoint0023.pth'
    config=checkpoint.parent/'config_cfg.py'
    report={'stage':a.stage,'started_at':datetime.now(timezone.utc).isoformat(),'runs':[]}
    reportfile=out/(a.stage+'_stage.json')
    if reportfile.exists(): raise FileExistsError(reportfile)
    def save(): reportfile.write_text(json.dumps(report,indent=2),encoding='utf-8')
    def run(name,fp32=False,proposals=True,raw=False):
        cmd=[sys.executable,'-u',str(root/'tools/legacy24_diagnosis.py'),'--mode','evaluate','--checkpoint',str(checkpoint),
             '--config',str(config),'--data-root','/workspace/DQDetr/data/path/AITODv2','--output-dir',str(out/name),
             '--image-ids',str(out/('precheck_ids.json' if raw else 'subset_ids.json')),'--group','D0' if raw else a.stage]
        if fp32: cmd+=['--fp32']
        if proposals: cmd+=['--export-proposals']
        if raw: cmd+=['--capture-raw']
        record={'name':name,'command':cmd,'started':time.time()}; report['runs'].append(record); save()
        print('RUN',name,flush=True)
        with (out/(name+'_console.log')).open('x',encoding='utf-8') as f:
            child=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
            record['pid']=child.pid; save(); warned=False
            while child.poll() is None:
                time.sleep(10)
                if time.time()-record['started']>7200 and not warned:
                    print('WARNING: group exceeds 2 hours; not terminating',flush=True); warned=True
            record.update(exit_code=child.returncode,ended=time.time()); save()
        if child.returncode: raise RuntimeError(f'{name} failed: {child.returncode}; no retry')
    try:
        if a.stage=='precheck':
            report['comparisons']={}
            for precision in ['fp32','amp']:
                for enabled in [False,True]: run(precision+('_on' if enabled else '_off'),fp32=precision=='fp32',proposals=enabled,raw=True)
                report['comparisons'][precision]=compare(out/(precision+'_off'),out/(precision+'_on'),read(out/'precheck_ids.json')); save()
        else:
            if read(out/'precheck_stage.json').get('status')!='completed': raise ValueError('Precheck has not passed')
            if a.stage=='D1' and read(out/'D0_stage.json').get('status')!='completed': raise ValueError('D0 incomplete')
            if a.stage=='D2':
                if not read(out/'decision.json').get('run_D2'): raise ValueError('D2 not justified by diagnostics')
            run(a.stage)
        report['status']='completed'; save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc)); save(); raise

if __name__=='__main__': main()
