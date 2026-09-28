"""Synthetic regression plus CPU historical initialization check; never reads dataset images."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def initialization(source,out):
    import torch
    import torchvision
    from unittest.mock import patch
    from main import build_model_main
    from util.slconfig import SLConfig
    from util.legacy_joint import initialize
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    rows=[]
    for name in ('control','h1_24e'):
        args=argparse.Namespace(**SLConfig.fromfile(str(ROOT/f'configs/legacy_joint/{name}.py'))._cfg_dict.to_dict())
        args.device='cpu';args.distributed=False;args.seed=42;args.pretrain_model_path=source
        directory=out/name;directory.mkdir(exist_ok=False);args.output_dir=str(directory)
        torch.manual_seed(42)
        with patch('torchvision.models.resnet50',side_effect=no_download):model,criterion,_=build_model_main(args)
        report=initialize(model,args)
        rows.append(dict(variant=name,common_sha256=model.joint_common_sha,coverage=report['coverage_by_numel']))
        del model,criterion
    if rows[0]['common_sha256']!=rows[1]['common_sha256']:
        raise RuntimeError('Historical common initialization differs')
    return rows


def main():
    from tools.benchmark_control import unique_dir,save
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained',required=True)
    p.add_argument('--output-dir',default=str(ROOT/'outputs/legacy_joint/validation'))
    args=p.parse_args();out=unique_dir(args.output_dir)
    print('Synthetic output:',out,flush=True)
    command=[sys.executable,'-u','-m','pytest','tests/test_legacy_joint.py','tests/test_joint_integration.py',
        'tests/test_loss_masking_repairs.py','tests/test_dn_repairs.py','tests/test_checkpoint_migration.py',
        '-q','-s','-ra',f'--junitxml={out / "tests.xml"}']
    with (out/'console.log').open('x',encoding='utf-8') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    report=dict(test_exit_code=code,gpu_environment=os.environ.get('CUDA_VISIBLE_DEVICES'),
        status='FAILED' if code else 'SYNTHETIC_PASSED',real_data_training='NOT_RUN',benchmark='NOT_RUN')
    try:
        if code:raise RuntimeError('Regression failed; no initialization certification')
        report['initialization']=initialization(args.pretrained,out)
        report['status']='SYNTHETIC_AND_INITIALIZATION_PASSED'
    except Exception as error:
        report.update(status='FAILED',error=repr(error));raise
    finally:save(out/'result.json',report)


if __name__=='__main__':main()
