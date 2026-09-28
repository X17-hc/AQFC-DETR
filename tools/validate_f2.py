"""Synthetic validation + frozen32 F0/F2 initialization checks. Never real training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from tools.check_p2_transfer import evaluate, compare_exports
    from tools.audit_encoder_stages import digest
    from util.p2_transfer import SOURCE_SHA
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--precheck-ids',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--synthetic-only',action='store_true')
    cli=p.parse_args()
    out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',command=sys.argv,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                checks={},limitation='No real training or full-test acceptance')
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Validation output:',out,flush=True);save()
    try:
        command=[sys.executable,'-u','-m','pytest','tests/test_f2.py','tests/test_precision24.py',
                 'tests/test_p2_transfer.py','-q','-s','-ra',f'--junitxml={out / "tests.xml"}']
        with (out/'synthetic.log').open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report['synthetic_exit_code']=result.returncode;save()
        if result.returncode: raise RuntimeError('Synthetic validation failed; inspect synthetic.log, no automatic retry')
        if not cli.synthetic_only:
            inputs=[Path(cli.pretrained),Path(cli.precheck_ids),Path(cli.data_root)/'annotations/aitodv2_test.json']
            hashes={str(p):digest(p) for p in inputs}
            if hashes[cli.pretrained]!=SOURCE_SHA: raise ValueError('Wrong S2 checkpoint')
            ids=json.loads(Path(cli.precheck_ids).read_text())
            known={v['id'] for v in json.loads(inputs[2].read_text())['images']}
            if len(ids)!=32 or len(set(ids))!=32 or any(type(v) is not int or v not in known for v in ids):
                raise ValueError('Requires frozen32 valid unique image IDs')
            report['input_hashes']=hashes
            for amp in [False,True]:
                cli.amp=amp
                tag='amp' if amp else 'fp32'
                a=evaluate(cli,out,'f0_'+tag,ids,'configs/precision24/f0_24e.py')
                b=evaluate(cli,out,'f2_'+tag,ids,'configs/precision24/f2_24e.py')
                comparison=compare_exports(out/('f0_'+tag),out/('f2_'+tag))
                report['checks'][tag]=dict(reference=a,f2=b,export_equivalence=comparison);save()
            if hashes!={str(p):digest(p) for p in inputs}: raise RuntimeError('Input hashes changed')
            report['inputs_unchanged']=True
        report['status']='PASSED_SYNTHETIC_ONLY' if cli.synthetic_only else 'PASSED_INITIALIZATION_NOT_TRAINING'
    except BaseException as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:save()
    print(report['status'],flush=True)


if __name__=='__main__':main()
