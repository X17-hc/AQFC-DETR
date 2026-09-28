"""Two-image causal probe only; hybrid reference is NOT a deployable model."""
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
from datasets import build_dataset
from util.incremental_checkpoint import validate_warmstart
from util.precision24 import NEW_PREFIXES
from util.misc import nested_tensor_from_tensor_list
from util.factor_diagnostics import image_diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', required=True)
    cli = p.parse_args(); root=Path(cli.audit_dir)
    dest=root/'transfer_probe.json'
    if dest.exists(): raise FileExistsError(dest)
    args=argparse.Namespace(**json.loads((root/'f1_6/config_args_all.json').read_text()))
    args.device='cuda'
    torch.manual_seed(42)
    original_resnet=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original_resnet(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,_,_=build_model_main(args)
    validate_warmstart(model,args)
    state=torch.load(args.pretrain_model_path,map_location='cpu',weights_only=False)['model']
    incompatible=model.load_state_dict(state,strict=False)
    assert not incompatible.unexpected_keys and all(k.startswith(NEW_PREFIXES) for k in incompatible.missing_keys)
    model.cuda().eval()
    dataset=build_dataset(args.eval_split,args)
    ids=json.loads((root/'image_ids.json').read_text())[:2]
    dataset.ids=ids
    encoder=model.transformer.encoder
    original=encoder.forward
    report={'images':[], 'warning':'hybrid reference uses extra six-layer compute; diagnosis only, not a fix'}
    cache={}
    def capture(src,*a,**kw):
        out=original(src,*a,**kw)
        cache['six']=out[0].clone()
        return out
    def hybrid(src,*a,**kw):
        out=original(src,*a,**kw)
        # The first level length is unambiguously the level_start_index[1].
        shapes=kw.get('spatial_shapes')
        if shapes is None:
            # Encoder positional signature: src, pos, spatial_shapes, ...
            shapes=a[1]
        cut=int(shapes[0].prod())
        memory=torch.cat((cache['six'][:,:cut],out[0][:,cut:]),1)
        return (memory,*out[1:])
    with torch.inference_mode():
        for idx in range(2):
            image,target=dataset[idx]
            sample=nested_tensor_from_tensor_list([image]).to('cuda')
            target={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in target.items()}
            row={'image_id':int(target['image_id'])}
            for name,depth,fn in [('six',6,capture),('two',2,original),('two_with_six_P2',2,hybrid)]:
                encoder.detail_full_layers=depth; encoder.forward=fn
                with torch.autocast('cuda',enabled=True): outputs=model(sample)
                row[name]=image_diagnostics(outputs,target,0)
            report['images'].append(row)
            print(json.dumps(row),flush=True)
    encoder.forward=original
    dest.write_text(json.dumps(report,indent=2))
    print(f'Completed diagnostic intervention: {dest}',flush=True)


if __name__=='__main__': main()
