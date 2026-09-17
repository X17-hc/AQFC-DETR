"""Exact-index differential regression; no datasets or checkpoint required."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('spatial_under_test', ROOT/'models/aqfcdetr/spatial_selection.py')
spatial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spatial)


def old_additions(ordered, remaining, cells, quotas):
    parts = [ordered[remaining[ordered] & (cells[ordered] == cell)][:quota]
             for cell, quota in enumerate(quotas) if quota]
    return torch.cat(parts) if parts else ordered[:0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('seed', range(12))
def test_segmented_selection_exact(device, seed):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    g = torch.Generator().manual_seed(seed)
    size = 127 + seed * 13
    cells = torch.randint(0, 24, (size,), generator=g).to(device)
    remaining = (torch.rand(size, generator=g) > .25).to(device)
    ordered = torch.randperm(size, generator=g).to(device)
    quotas = torch.randint(0, 30, (24,), generator=g).tolist()
    expected = old_additions(ordered, remaining, cells, quotas)
    actual = spatial._quota_additions(ordered, remaining, cells, quotas)
    assert torch.equal(actual, expected)
    assert actual.unique().numel() == actual.numel()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('case', ['random', 'ties', 'zero', 'nan', 'capacity', 'ratio0', 'ratio1'])
def test_full_indices_reports_and_gather_gradients(device, case):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    shapes = torch.tensor([[17, 23], [9, 12], [5, 6]], device=device)
    size = 17*23 + 9*12 + 5*6
    semantic = torch.randn(2, size, device=device)
    density = torch.rand(2, size, device=device)
    padding = torch.zeros(2, size, device=device, dtype=torch.bool)
    padding[0, :17*23].reshape(17,23)[-3:, -5:] = True
    invalid = padding.clone(); invalid[1, ::3] = True
    if case == 'ties': semantic.zero_(); density.fill_(.5)
    if case == 'zero': density.zero_()
    if case == 'nan': density[0, 10] = float('nan')
    if case == 'capacity': invalid[0, 7:] = True
    ratio = 0. if case == 'ratio0' else 1. if case == 'ratio1' else .75
    a, b = [], []
    args = (semantic, density, padding, invalid, 300, shapes)
    with patch.object(spatial, '_quota_additions', old_additions):
        expected = spatial.spatial_indices(*args, ratio=ratio, grid=(3,8), diagnostics=a)
    actual = spatial.spatial_indices(*args, ratio=ratio, grid=(3,8), diagnostics=b)
    assert torch.equal(expected, actual)
    assert a == b
    memory = torch.randn(2, size, 4, device=device, requires_grad=True)
    reference = memory.detach().clone().requires_grad_()
    memory.gather(1, actual[..., None].expand(-1,-1,4)).square().sum().backward()
    reference.gather(1, expected[..., None].expand(-1,-1,4)).square().sum().backward()
    assert torch.equal(memory.grad, reference.grad)


def test_empty_quotas_and_exhausted_candidates():
    order=torch.arange(8); cells=order%3; remaining=torch.zeros(8,dtype=torch.bool)
    assert spatial._quota_additions(order,remaining,cells,[5,5,5]).numel()==0
    assert spatial._quota_additions(order,~remaining,cells,[0,0,0]).numel()==0


@pytest.mark.parametrize('amp',[False,True])
def test_real_model_reference_vs_optimized_forward_backward(amp,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('Native CUDA integration')
    import argparse
    import copy
    import sys
    import torchvision
    sys.path.insert(0,str(ROOT))
    from main import build_model_main
    from util.slconfig import SLConfig
    from models.aqfcdetr import spatial_selection as production
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',False)
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',True)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    config=SLConfig.fromfile(str(ROOT/'configs/experiments/local8gb/light_spatial.py'))._cfg_dict.to_dict()
    config.update(device='cuda',distributed=False,use_dn=True,dn_number=4,
                  classification_loss_type='quality_blend')
    original=torchvision.models.resnet50
    def no_download(*args,**kwargs):
        kwargs['weights']=None
        return original(*args,**kwargs)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,criterion,_=build_model_main(argparse.Namespace(**config))
    model.cuda().train(); criterion.cuda(); criterion.quality_lambda=.25
    model.set_epoch(11)
    model.transformer.force_query_budget=16
    state=copy.deepcopy(model.state_dict())
    images=[torch.rand(3,128,128,device='cuda')]
    targets=[dict(boxes=torch.tensor([[.4,.5,.1,.1]],device='cuda'),labels=torch.tensor([0],device='cuda'))]
    snapshots=[]
    for implementation in (production._quota_additions_reference,production._quota_additions_reference,
                           production._quota_additions):
        model.load_state_dict(state);model.zero_grad(set_to_none=True)
        torch.manual_seed(91);torch.cuda.manual_seed_all(91)
        with patch.object(production,'_quota_additions',implementation), torch.amp.autocast('cuda',enabled=amp):
            output=model(images,targets)
            losses=criterion(output,targets)
            loss=sum(value*criterion.weight_dict[key] for key,value in losses.items() if key in criterion.weight_dict)
        assert torch.isfinite(loss)
        loss.backward()
        snapshots.append(dict(logits=output['pred_logits'].detach().cpu(),boxes=output['pred_boxes'].detach().cpu(),
            loss=loss.detach().cpu(),grads={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}))
        del output,loss,losses
    for key in ('logits','boxes','loss'):
        torch.testing.assert_close(snapshots[0][key],snapshots[1][key],atol=0,rtol=0)
        torch.testing.assert_close(snapshots[0][key],snapshots[2][key],atol=0,rtol=0)
    assert snapshots[0]['grads'].keys()==snapshots[1]['grads'].keys()
    # FP32 keeps the agreed tolerance. AMP atomics are audited separately against
    # a reference/reference repeat, not silently judged with a widened tolerance.
    differences=[]
    for name in snapshots[0]['grads']:
        a,b,c=(s['grads'][name] for s in snapshots)
        assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(c).all(),name
        differences.append((name,float((a-b).abs().max()),float((a-c).abs().max())))
    import json
    print('GRADIENT_AUDIT',json.dumps(dict(amp=amp,fields=['name','reference_repeat_max_abs','optimized_max_abs'],
        largest=sorted(differences,key=lambda r:-max(r[1:]))[:8])))
    if not amp:
        for name in snapshots[0]['grads']:
            torch.testing.assert_close(snapshots[0]['grads'][name],snapshots[2]['grads'][name],
                                       atol=1e-5,rtol=1e-4)
