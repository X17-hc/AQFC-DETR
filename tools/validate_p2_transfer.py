"""Authorized synthetic tests plus frozen 32-image reference compatibility; never trains real data."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','pretrained','precheck-ids','output-dir'): p.add_argument('--'+name,required=True)
    cli=p.parse_args()
    out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    commands=[[sys.executable,'-u','-m','pytest','tests/test_p2_transfer.py','tests/test_precision24.py',
               'tests/test_encoder_stage_audit.py','-q','-s',f'--junitxml={out / "tests.xml"}']]
    for precision in ('--no-amp','--amp'):
        commands.append([sys.executable,'-u',str(ROOT/'tools/check_p2_transfer.py'),
            '--mode','reference-check','--data-root',cli.data_root,'--pretrained',cli.pretrained,
            '--precheck-ids',cli.precheck_ids,'--output-dir',str(out/precision[2:]),precision])
    print(f'Output: {out}',flush=True)
    results=[]
    for i,command in enumerate(commands):
        with (out/f'{i}.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            code=process.wait()
        results.append(dict(command=command,exit_code=code))
        (out/'results.json').write_text(json.dumps(results,indent=2))
        if code: raise SystemExit(code)
    print('Synthetic and reference checks complete. Student quality still requires manual adaptation and acceptance.',flush=True)


if __name__=='__main__': main()
