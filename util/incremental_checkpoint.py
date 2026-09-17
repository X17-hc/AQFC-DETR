"""Strict model-only warm-start; optimizer and schedule are intentionally fresh."""
import torch
from .experiment import sha256


def validate_warmstart(model,args):
    expected=getattr(args,'expected_pretrained_sha256','')
    if expected and sha256(args.pretrain_model_path)!=expected:
        raise ValueError('Warm-start SHA-256 differs from the frozen checkpoint')
    checkpoint=torch.load(args.pretrain_model_path,map_location='cpu',weights_only=False)
    metadata=checkpoint.get('run_metadata',{})
    epoch=getattr(args,'expected_pretrained_epoch',None)
    if epoch is not None and checkpoint.get('epoch')!=epoch:
        raise ValueError('Warm-start checkpoint epoch differs from the approved epoch')
    if metadata.get('epoch_complete') is not True or metadata.get('smoke_test'):
        raise ValueError('Strict warm-start requires an explicitly complete epoch checkpoint')
    saved=checkpoint.get('model',{})
    target=model.state_dict()
    if saved.keys()!=target.keys() or any(saved[k].shape!=target[k].shape for k in target):
        raise ValueError('Strict warm-start requires 100% model state coverage')
    if any(not torch.isfinite(v).all() for v in saved.values() if v.is_floating_point()):
        raise ValueError('Non-finite state in warm-start checkpoint')
    # A matching tensor shape alone does not establish matching module semantics.
    signature=checkpoint.get('variant_signature',{})
    for key in ('allocator_encoder_type','allocator_enabled','calibrator_enabled','proposal_selection_mode',
                'query_budget_levels','calibrator_gate_type','calibrator_use_spatial'):
        if signature.get(key)!=getattr(args,key,None):
            raise ValueError(f'Warm-start architecture mismatch: {key}')
