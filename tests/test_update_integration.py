"""Synthetic only: no dataset files, no checkpoint downloads, no real training."""
from unittest.mock import patch
import argparse
import pytest
import torch
import torchvision
from util.slconfig import SLConfig
from main import build_model_main
from models.aqfcdetr.query_allocator import QueryBudgetLoss


@pytest.mark.parametrize('variant', ['baseline', 'light_spatial', 'modules_off_900'])
def test_synthetic_model_backward_and_optimizer(variant, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA native attention integration')
    torch.manual_seed(42)
    args = SLConfig.fromfile(f'configs/experiments/local8gb/{variant}.py')._cfg_dict.to_dict()
    args.update(device='cuda', distributed=False)
    original = torchvision.models.resnet50
    def no_download(*a, **kw):
        kw['weights'] = None
        return original(*a, **kw)
    with patch('torchvision.models.resnet50', side_effect=no_download):
        model, criterion, post = build_model_main(argparse.Namespace(**args))
    model.cuda().train(); criterion.cuda()
    # Five feature levels require >1 spatial value in the existing DGFC BatchNorm.
    # 64px yields a 1x1 final level at batch=1; 128px preserves the normal contract.
    image = torch.rand(3, 128, 128, device='cuda')
    target = dict(boxes=torch.tensor([[.5,.5,.1,.1]], device='cuda'),
                  labels=torch.tensor([0], device='cuda'), size=torch.tensor([128,128],device='cuda'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    with torch.amp.autocast('cuda'):
        output = model([image], [target])
        losses = criterion(output, [target])
        loss = sum(v * criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
        if args['allocator_enabled']:
            budget = QueryBudgetLoss(density_target_backend=args['density_target_backend']).cuda()
            loss += budget(output['allocator_outputs'], dict(real_counts=torch.tensor([1.],device='cuda'),
                                                            targets=[target]))['loss_allocator_total']
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
    optimizer.step()
    if not args['allocator_enabled']:
        assert model.transformer.query_allocator is None
        assert model.transformer.feature_calibrator is None
        assert not any('query_allocator' in n or 'feature_calibrator' in n for n,_ in model.named_parameters())
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        result = model([image])
        boxes = post['bbox'](result, torch.tensor([[128,128]],device='cuda'))
    assert len(boxes) == 1
    assert torch.isfinite(boxes[0]['boxes']).all()
    if variant == 'light_spatial':
        model.transformer.force_query_budget = 900
        with torch.no_grad(), torch.amp.autocast('cuda'):
            model.transformer.grouped_decoder_inference = True
            grouped = model([image])
            model.transformer.grouped_decoder_inference = False
            padded = model([image])
        torch.testing.assert_close(grouped['pred_logits'], padded['pred_logits'], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(grouped['pred_boxes'], padded['pred_boxes'], atol=1e-5, rtol=1e-4)
    from util.checkpoint import load_native_resume
    checkpoint = tmp_path / 'synthetic_native.pth'
    torch.save(dict(model=model.state_dict(), epoch=0), checkpoint)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1)
    load_native_resume(model, checkpoint)
    torch.testing.assert_close(parameter, before, atol=0, rtol=0)
    del optimizer, model, criterion


@pytest.mark.parametrize('variant', ['baseline', 'aqba_light'])
@pytest.mark.parametrize('filename', ['dqdetr_best305.pth', 'pretrain_model.pth'])
def test_existing_weights_warmstart_variants(variant, filename, tmp_path):
    from pathlib import Path
    from util.checkpoint_migration import load_legacy_pretrained
    source = Path('weights/legacy') / filename
    if not source.exists():
        pytest.skip('Local legacy weight artifact unavailable')
    args = SLConfig.fromfile(f'configs/experiments/local8gb/{variant}.py')._cfg_dict.to_dict()
    args.update(device='cpu', distributed=False)
    original = torchvision.models.resnet50
    def no_download(*a, **kw):
        kw['weights'] = None
        return original(*a, **kw)
    with patch('torchvision.models.resnet50', side_effect=no_download):
        model, _, _ = build_model_main(argparse.Namespace(**args))
    report = load_legacy_pretrained(model, source, tmp_path/'migration.json')
    assert report['parameter_only_coverage'] > .5
    if variant == 'aqba_light':
        assert any('light_density_encoder' in key for key in report['missing_keys'])
    assert (tmp_path/'migration.json').is_file()
