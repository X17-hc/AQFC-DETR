"""CPU-only source/initialization audit; no image reads, training or inference."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import build_model_main
from util.slconfig import SLConfig
from util.config_validation import validate_config
from util.incremental_checkpoint import validate_warmstart
from util.experiment import sha256
from util.precision24 import NEW_PREFIXES, param_groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    source=torch.load(args.pretrained,map_location='cpu',weights_only=False)
    report=dict(source=args.pretrained,sha256=sha256(args.pretrained),epoch=source.get('epoch'),
                run_metadata=source.get('run_metadata'),variants={})
    for arm in ('f0','f1'):
        values=SLConfig.fromfile(str(ROOT/f'configs/precision24/{arm}_24e.py'))._cfg_dict.to_dict()
        validate_config(values)
        values.update(device='cpu',distributed=False,pretrain_model_path=args.pretrained)
        config=argparse.Namespace(**values)
        torch.manual_seed(42)
        with patch('torchvision.models.resnet50',side_effect=no_download):
            model,_,_=build_model_main(config)
        validate_warmstart(model,config)
        missing=model.load_state_dict(source['model'],strict=arm=='f0')
        assert not missing.unexpected_keys and all(k.startswith(NEW_PREFIXES) for k in missing.missing_keys)
        loaded=model.state_dict()
        assert all(torch.equal(loaded[k],value) for k,value in source['model'].items())
        groups=param_groups(config,model)
        report['variants'][arm]=dict(shared_state_equal=True,new_keys=missing.missing_keys,
            parameters=sum(p.numel() for p in model.parameters()),
            groups=[dict(name=g['precision24_group'],lr=g['lr'],
                         parameter_count=sum(p.numel() for p in g['params'])) for g in groups])
        del groups,loaded,model
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
