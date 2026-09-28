"""Version, initialization, freeze and acceptance contracts for P2 layerwise migration."""
import hashlib
import json
from pathlib import Path
import torch

VERSION = 'layerwise_v2'
SOURCE_SHA = '40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928'
PREFIX = 'transformer.encoder.p2_substitutes.'
DEFAULTS = dict(p2_transfer_version=None,p2_transfer_mode='reference')


def validate(config, errors):
    for key in config:
        if key.startswith('p2_transfer_') and key not in DEFAULTS:
            errors.append(f'Unknown P2 transfer config: {key}')
    version = config.get('p2_transfer_version')
    if version is None:
        if config.get('p2_transfer_mode','reference') != 'reference':
            errors.append('P2 student mode requires layerwise_v2')
        return
    if version != VERSION or config.get('p2_transfer_mode') not in ('reference','student'):
        errors.append('Unsupported P2 transfer version/mode')
    if not config.get('precision24_enabled') or config.get('detail_context_encoder'):
        errors.append('P2 transfer requires precision24 and disables terminal detail fusion')
    if not config.get('local_refiner_enabled') or config.get('precision24_stage_override') is not None:
        errors.append('P2 transfer requires local refiner and no old stage override')


def configure(model,args):
    if getattr(args,'p2_transfer_version',None) is None:
        return
    from models.aqfcdetr.p2_transfer import P2Substitute
    from torch import nn
    encoder = model.transformer.encoder
    if hasattr(encoder,'detail_fusion'):
        raise ValueError('Terminal fusion must not coexist with layerwise substitutes')
    with torch.random.fork_rng(devices=[]):
        encoder.p2_substitutes = nn.ModuleList([P2Substitute(model.hidden_dim) for _ in range(4)])
    encoder.p2_transfer_mode = args.p2_transfer_mode
    model.p2_transfer_version = VERSION


def signature(args):
    values = args if isinstance(args,dict) else vars(args)
    if values.get('p2_transfer_version') is None: return {}
    return {k:values.get(k,v) for k,v in DEFAULTS.items()}


def freeze_for_adaptation(model):
    model.eval()
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(PREFIX))
    model.transformer.encoder.p2_substitutes.train()


def frozen_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith(PREFIX): continue
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract_inputs(model,samples):
    """Run frozen backbone/projectors exactly once, stopping BEFORE encoder/AQBA/decoder."""
    captured = []
    class Captured(Exception): pass
    def hook(module,args,kwargs):
        src = args[0] if args else kwargs['src']
        captured.append((src,kwargs['pos'],kwargs['spatial_shapes'],kwargs['level_start_index'],
                         kwargs['valid_ratios'],kwargs['key_padding_mask']))
        raise Captured()
    handle = model.transformer.encoder.register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad(): model(samples)
    except Captured:
        pass
    finally:
        handle.remove()
    if len(captured) != 1: raise RuntimeError('Failed to capture frozen encoder inputs')
    return captured[0]


def load_artifact(model,filename,require_receipt=False,receipt=None):
    from .experiment import sha256
    data=torch.load(filename,map_location='cpu',weights_only=False)
    if (data.get('kind')!='p2_adapter_initialization' or data.get('version')!=VERSION or
            data.get('source_sha256')!=SOURCE_SHA or not data.get('complete_two_epochs')):
        raise ValueError('Requires complete two-epoch layerwise_v2 adapter initialization')
    if require_receipt:
        r=json.loads(Path(receipt).read_text()) if receipt else {}
        if (r.get('accepted') is not True or r.get('adapter_sha256')!=sha256(filename) or
                r.get('source_sha256')!=SOURCE_SHA or r.get('image_count')!=1000):
            raise ValueError('Formal student training requires matching 1000-image acceptance receipt')
    state=data['adapters']
    if any(not torch.isfinite(x).all() for x in state.values() if x.is_floating_point()):
        raise ValueError('Non-finite adapter state')
    model.transformer.encoder.p2_substitutes.load_state_dict(state,strict=True)
    return data


def initialize_student(model,args):
    active=getattr(args,'p2_transfer_version',None)==VERSION
    filename=getattr(args,'p2_adapter','')
    if not active:
        if filename: raise ValueError('--p2-adapter requires a P2 transfer model')
        return
    if getattr(args,'resume',''):
        if filename: raise ValueError('Do not combine --resume and --p2-adapter')
        return
    if args.p2_transfer_mode=='student':
        if not filename: raise ValueError('Student entry requires --p2-adapter; random student cannot start training')
        load_artifact(model,filename,require_receipt=not args.eval,
                      receipt=getattr(args,'p2_acceptance',''))
    elif not args.eval:
        raise ValueError('Reference is evaluation-only; use dedicated adaptation trainer')
    elif filename:
        raise ValueError('Reference mode must not load a student adapter')
