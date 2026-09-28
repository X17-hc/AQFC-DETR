"""MANUAL acceptance, or automatic reference-only compatibility. Never trains."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_encoder_stages import digest,check_exports
from tools.legacy24_diagnosis import sample
from util.p2_transfer import SOURCE_SHA,VERSION


def summarize(directory,ids):
    rows=check_exports(directory,ids)
    for row in rows:
        if row['query_mask_count']!=row['executed_query_count']:
            raise ValueError('Invalid query mask in export')
        if row['predicted_count'] is None or not math.isfinite(row['predicted_count']):
            raise ValueError('Invalid allocator count')
    metrics=json.loads((directory/'metrics.jsonl').read_text().splitlines()[-1])
    stats=metrics['test_coco_eval_bbox']
    if min(stats[0],stats[4])<0 or not all(math.isfinite(stats[i]) for i in (0,4)):
        raise ValueError('AP/APvt support is unavailable; no quality pass possible')
    gt=sum(r['gt_count'] for r in rows)
    if not gt: raise ValueError('No GT support')
    return dict(AP=100*stats[0],APvt=100*stats[4],
        proposal_050=100*sum(r['proposal_hits_050'] for r in rows)/gt,
        count_mae=sum(abs(r['predicted_count']-r['gt_count']) for r in rows)/len(rows),
        mean_queries=sum(r['executed_query_count'] for r in rows)/len(rows),
        images=len(rows),gt=gt)


def gate(reference,student):
    checks=dict(AP=student['AP']>=reference['AP']-1.,
        APvt=student['APvt']>=reference['APvt']-1.,
        proposal_050=student['proposal_050']>=reference['proposal_050']-2.,
        count_mae=student['count_mae']<=reference['count_mae']+max(1.,.1*reference['count_mae']),
        mean_queries=student['mean_queries']<=1.2*reference['mean_queries'])
    return dict(accepted=all(checks.values()),checks=checks,reference=reference,student=student)


def evaluate(cli,out,name,ids,config,adapter=''):
    target=out/name; target.mkdir()
    idfile=target/'ids.json'; idfile.write_text(json.dumps(ids))
    command=[sys.executable,'-u',str(ROOT/'main.py'),'--config',str(ROOT/config),
        '--pretrained',cli.pretrained,'--data-root',cli.data_root,'--output-dir',str(target),
        '--eval','--device','cuda','--num_workers','0','--seed','42','--eval-image-ids',str(idfile),
        '--export-predictions','--export-diagnostics']
    command+=['--amp' if cli.amp else '--no-amp']
    if adapter: command+=['--p2-adapter',adapter]
    (target/'command.json').write_text(json.dumps(command,indent=2))
    print(f'Evaluating {name}: {len(ids)} images; log={target}/console.log',flush=True)
    with (target/'console.log').open('x') as log:
        proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode: raise RuntimeError(f'{name} exit={proc.returncode}; no automatic retry')
    resolved=json.loads((target/'config_args_all.json').read_text())
    if resolved['amp']!=cli.amp:
        raise RuntimeError('Actual child AMP configuration differs from requested precision')
    return summarize(target,ids)


def compare_exports(a,b):
    a=json.loads(next(a.rglob('predictions.json')).read_text())
    b=json.loads(next(b.rglob('predictions.json')).read_text())
    if len(a)!=len(b) or any((x['image_id'],x['category_id'])!=(y['image_id'],y['category_id']) for x,y in zip(a,b)):
        raise AssertionError('Reference prediction identity differs')
    x=np.array([[v['score'],*v['bbox']] for v in a]).reshape(-1,5)
    y=np.array([[v['score'],*v['bbox']] for v in b]).reshape(-1,5)
    np.testing.assert_allclose(x,y,atol=1e-5,rtol=1e-4)
    return dict(max_abs=float(np.max(np.abs(x-y))) if x.size else 0.,atol=1e-5,rtol=1e-4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['reference-check','accept'],required=True)
    p.add_argument('--data-root',required=True); p.add_argument('--pretrained',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--precheck-ids',required=True)
    p.add_argument('--adapter',default='')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    cli=p.parse_args()
    if digest(cli.pretrained)!=SOURCE_SHA: raise ValueError('S2 source hash differs')
    if cli.mode=='accept' and not Path(cli.adapter).is_file():
        raise ValueError('Set --adapter to the completed two-epoch adapter_initialization.pth')
    out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    annotation=Path(cli.data_root)/'annotations/aitodv2_test.json'
    raw=json.loads(annotation.read_text()); known={x['id'] for x in raw['images']}
    ids=json.loads(Path(cli.precheck_ids).read_text())
    if len(ids)!=32 or len(set(ids))!=32 or any(type(i) is not int or i not in known for i in ids):
        raise ValueError('Requires exact frozen 32-image precheck')
    report=dict(version=VERSION,command=sys.argv,source_sha256=SOURCE_SHA,
        annotation_sha256=digest(annotation),precheck_sha256=digest(cli.precheck_ids),amp=cli.amp,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),mode=cli.mode,
        adapter_sha256=digest(cli.adapter) if cli.adapter else None,stages=[],accepted=False)
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'Output: {out}',flush=True); save()
    try:
        if cli.mode=='reference-check':
            a=evaluate(cli,out,'f0',ids,'configs/precision24/f0_24e.py')
            b=evaluate(cli,out,'reference',ids,'configs/p2_transfer/reference.py')
            report.update(reference_compatibility=compare_exports(out/'f0',out/'reference'),
                reference_metrics=a,transfer_reference_metrics=b,status='REFERENCE_COMPATIBLE_ONLY')
        else:
            for count in (32,1000):
                if count==1000:
                    ids,distribution=sample(raw,1000); report['distribution']=distribution
                a=evaluate(cli,out,f'reference_{count}',ids,'configs/p2_transfer/reference.py')
                b=evaluate(cli,out,f'student_{count}',ids,'configs/p2_transfer/student_24e.py',cli.adapter)
                result=gate(a,b); report['stages'].append(result); save()
                print(json.dumps(result),flush=True)
                if not result['accepted']:
                    report['status']='QUALITY_GATE_FAILED'; save()
                    raise SystemExit(2)
            report.update(image_count=1000,status='SUBSET_ACCEPTED_NOT_FULL_TEST')
        if digest(cli.pretrained)!=SOURCE_SHA or digest(annotation)!=report['annotation_sha256']:
            raise RuntimeError('Input files changed')
        if digest(cli.precheck_ids)!=report['precheck_sha256']:
            raise RuntimeError('Precheck list changed')
        if cli.adapter and digest(cli.adapter)!=report['adapter_sha256']:
            raise RuntimeError('Adapter changed during acceptance')
        if cli.mode=='accept':
            # Issue permission for formal training only after all input invariants pass.
            report['accepted']=True
            (out/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        save(); print(report['status'],flush=True)
    except BaseException:
        report['accepted']=False
        if report.get('status')!='QUALITY_GATE_FAILED': report['status']='EXECUTION_FAILED'
        save(); raise


if __name__=='__main__': main()
