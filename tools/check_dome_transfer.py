"""Synthetic checks only. Does not read training images or initialize CUDA."""
import os
import argparse
import json
from datetime import datetime,timezone
import uuid
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained')
    parser.add_argument('--output-dir',default=str(ROOT/'outputs/dome_transfer/check'))
    cli=parser.parse_args()
    out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    print('Output:',out,flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''  # This CPU-only test process, never a training binding.
    with (out/'console.log').open('x',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-m','pytest','-q',
            'tests/test_dome_transfer.py','tests/test_legacy_joint.py'],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    if code:raise SystemExit(code)
    if cli.pretrained:
        sys.path.insert(0,str(ROOT))
        import tempfile
        import torch
        from main import build_model_main
        from util.slconfig import SLConfig
        from util.config_validation import validate_config
        from util.dome_transfer import initialize
        from util.legacy_joint import tensor_digest
        digests=[]
        with tempfile.TemporaryDirectory(prefix='aqfc-dome-init-') as temp:
            for group in ('d0','d1'):
                values=SLConfig.fromfile(str(ROOT/f'configs/dome_transfer/{group}_3e.py'))._cfg_dict.to_dict()
                validate_config(values)
                args=argparse.Namespace(**dict(values,device='cpu',distributed=False,seed=42,
                    output_dir=temp,pretrain_model_path=cli.pretrained))
                model,criterion,_=build_model_main(args)
                report=initialize(model,args)
                digests.append(tensor_digest(model.state_dict()))
                assert criterion.joint_mature and report['coverage_by_numel']==1.
                del model,criterion
        if len(set(digests))!=1:raise RuntimeError('D0/D1 initial state differs')
        result=dict(strict_initialization='passed',state_sha256=digests[0],groups=['d0','d1'])
        (out/'initialization.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        with (out/'console.log').open('a',encoding='utf-8') as log:log.write(json.dumps(result)+'\n')
        print(json.dumps(result))
