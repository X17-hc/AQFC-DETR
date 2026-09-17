"""Synthetic/unit verification entry; never starts real-data training."""
import argparse
from pathlib import Path
import subprocess
import sys
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.slconfig import SLConfig
from util.config_validation import validate_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--junit',default='')
    args=p.parse_args()
    for file in sorted((ROOT/'configs/incremental_v2').glob('*.py')):
        validate_config(SLConfig.fromfile(str(file))._cfg_dict.to_dict())
        print(f'[Config OK] {file.name}',flush=True)
    command=[sys.executable,'-m','pytest','tests','-q','--tb=short','-p','no:cacheprovider']
    if args.junit:
        path=Path(args.junit)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        command+=['--junitxml',str(path.resolve())]
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
    subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__': main()
