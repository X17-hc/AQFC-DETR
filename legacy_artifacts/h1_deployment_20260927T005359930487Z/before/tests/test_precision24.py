"""Synthetic contract checks only; never read real images or detector weights."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from models.aqfcdetr.precision_modules import DetailContextFusion, LocalBoxRefiner
from util import precision24 as contract
from util.slconfig import SLConfig
from util.config_validation import validate_config


def recipe(name='f1_24e'):
    config = SLConfig.fromfile(f'configs/precision24/{name}.py')._cfg_dict.to_dict()
    validate_config(config)
    return argparse.Namespace(**config)


def test_recipes_and_validation():
    a, b = vars(recipe('f0_24e')), vars(recipe())
    assert {k for k in a if a[k] != b[k]} == {'detail_context_encoder', 'local_refiner_enabled'}
    assert a['epochs'] == 24 and a['val_epoch'] == [23]
    assert a['training_phase_fixed_epoch'] == 23 and a['expected_pretrained_epoch'] == 2
    assert a['dn_number'] == 100 and a['geometry_loss_weight'] == 0
    for key, value in [('hidden_dim', 128), ('precision24_stage_override', 3), ('precision24_new_lr', float('nan'))]:
        bad = dict(b, **{key: value})
        with pytest.raises(ValueError): validate_config(bad)


def test_fusion_identity_padding_and_gradients():
    module = DetailContextFusion(8)
    shapes = torch.tensor([[5, 7], [3, 4], [2, 2], [1, 1], [1, 1]])
    memory = torch.randn(2, 53, 8, requires_grad=True)
    mask = torch.zeros(2, 53, dtype=torch.bool)
    mask[0, 30:35] = True
    out = module(memory, shapes, mask)
    torch.testing.assert_close(out, memory, atol=0, rtol=0)
    out.square().mean().backward()
    assert module.output.weight.grad.abs().sum() > 0
    with torch.no_grad(): module.output.weight.fill_(.01)
    changed = module(memory, shapes, mask)
    torch.testing.assert_close(changed[mask], memory[mask], atol=0, rtol=0)
    torch.testing.assert_close(changed[:, 35:], memory[:, 35:], atol=0, rtol=0)


def local_inputs():
    torch.manual_seed(42)
    return (torch.randn(2, 5, 16, requires_grad=True),
            torch.tensor([[[.5,.5,.01,.02],[0.,0.,.1,.1],[1.,1.,.02,.03],[.3,.8,.1,.2],[.1,.2,.3,.2]]]*2,
                         requires_grad=True),
            [torch.randn(2,16,9,13,requires_grad=True), torch.randn(2,16,5,7,requires_grad=True)],
            [torch.zeros(2,9,13,dtype=torch.bool), torch.zeros(2,5,7,dtype=torch.bool)],
            torch.tensor([[33,49],[28,40]]),
            torch.tensor([[True]*5,[True,True,False,False,False]]))


def test_refiner_identity_two_steps_chunk_mask_and_bounds():
    args = local_inputs()
    module = LocalBoxRefiner(16, 8, chunk_size=2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.01)
    out = module(*args)
    torch.testing.assert_close(out, args[1], atol=0, rtol=0)
    out[args[-1]].sum().backward()
    assert module.head[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(args[1].grad).all()
    optimizer.step(); optimizer.zero_grad()
    fresh = local_inputs()
    out = module(*fresh)
    out[fresh[-1]].sum().backward()
    assert fresh[0].grad.abs().sum() > 0
    assert all(f.grad.abs().sum() > 0 for f in fresh[2])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert (fresh[0].grad[~fresh[-1]] == 0).all()
    torch.testing.assert_close(out[~fresh[-1]], fresh[1][~fresh[-1]])
    whole = copy.deepcopy(module); whole.chunk_size = 256
    torch.testing.assert_close(out, whole(*fresh), atol=1e-5, rtol=1e-4)
    ratio = out[...,2:] / fresh[1][...,2:]
    assert (ratio >= 1/1.5).all() and (ratio <= 1.5).all()


def test_sampling_detaches_coordinates_and_excludes_padding():
    module = LocalBoxRefiner(16, 8)
    feature = torch.ones(1,8,5,7,requires_grad=True)
    mask = torch.ones(1,5,7,dtype=torch.bool)
    boxes = torch.tensor([[[10.,10.,2.,2.]]],requires_grad=True)
    sampled = module.sample(feature, mask, boxes, 4, 1.5, 16.)
    assert torch.count_nonzero(sampled) == 0
    sampled.sum().backward()
    assert boxes.grad is None and torch.count_nonzero(feature.grad) == 0


def test_coarse_matching_equal_regression_and_refined_quality_target():
    from models.aqfcdetr.model import SetCriterion
    from models.aqfcdetr.matcher import HungarianMatcher
    matcher = HungarianMatcher()
    criterion = SetCriterion(2,matcher,{},.25,['labels','boxes'],aligned_box_loss=True,
                             classification_loss_type='quality_blend')
    criterion.quality_lambda = .25
    coarse = torch.tensor([[[.48,.5,.1,.1],[.8,.8,.1,.1]]],requires_grad=True)
    refined = torch.tensor([[[.5,.5,.1,.1],[.8,.8,.1,.1]]],requires_grad=True)
    logits = torch.tensor([[[2.,-1.],[99.,99.]]],requires_grad=True)
    out = dict(pred_logits=logits,pred_boxes=refined,pred_boxes_coarse=coarse,
               query_valid_mask=torch.tensor([[True,False]]),dn_meta=None)
    targets = [dict(boxes=torch.tensor([[.5,.5,.1,.1]]),labels=torch.tensor([0]))]
    calls = []
    def spy(outputs, targets):
        calls.append(outputs['pred_boxes'])
        return [(torch.tensor([0]),torch.tensor([0]))]
    with patch.object(matcher,'forward',side_effect=spy): result = criterion(out,targets)
    assert len(calls) == 1 and calls[0] is coarse
    indices = [(torch.tensor([0]),torch.tensor([0]))]
    fine_losses = criterion.loss_boxes(out,targets,indices,1)
    old_losses = criterion.loss_boxes(dict(out,pred_boxes=coarse),targets,indices,1)
    for key in fine_losses:
        torch.testing.assert_close(result[key],.5*(fine_losses[key]+old_losses[key]))
    expected = criterion.loss_labels(out,targets,indices,1)['loss_ce']
    torch.testing.assert_close(result['loss_ce'],expected)
    result['loss_ce'].backward(retain_graph=True)
    assert refined.grad is None and coarse.grad is None  # Quality target detached.
    assert (logits.grad[:,1] == 0).all()
    result['loss_bbox'].backward()
    assert coarse.grad is not None and refined.grad is not None


def test_new_launchers_are_exact_manual_recipes():
    import shlex
    import xml.etree.ElementTree as ET
    from main import get_args_parser
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'.run').glob('AQFC-DETR F[01]*.run.xml'))
    assert len(paths) == 6  # Includes F0 smoke, in addition to the requested six-entry set.
    for path in paths:
        cfg = ET.parse(path).getroot().find('configuration')
        options = {x.get('name'):x.get('value') for x in cfg.findall('option')}
        assert options['SDK_HOME'] == '/opt/conda/envs/AQFC-DETR/bin/python'
        assert options['WORKING_DIRECTORY'] == '/workspace/AQFC-DETR'
        argv = shlex.split(options['PARAMETERS'])
        if '性能测量' in path.name:
            from tools.benchmark_interleaved import parser
            args = parser().parse_args(argv)
            assert args.comparison == 'precision24' and args.order == 'f0,f1,f1,f0'
            assert not args.profile and args.warmup == 50 and args.steps == 200
        else:
            args = get_args_parser().parse_args(argv)
            assert not args.resume and args.unique_output_dir and args.amp
            assert args.max_train_steps == (20 if '20步' in path.name else 0)
            assert args.max_eval_steps == (2 if '20步' in path.name else 0)
            assert args.num_workers == 2
            assert args.pretrain_model_path.endswith('20260924_115851_063163_8e60e76c05c3/checkpoint0002.pth')


def test_schedule_optimizer_and_restore():
    model = nn.Module(); model.hidden_dim = 16
    model.backbone = nn.Linear(3,16)
    model.transformer = nn.Module(); model.transformer.encoder = nn.Module()
    args = recipe()
    rng = torch.get_rng_state().clone()
    contract.configure(model, args)
    assert torch.equal(rng, torch.get_rng_state())
    groups = contract.param_groups(args, model)
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    optimizer = torch.optim.AdamW(groups)
    for epoch, depth in [(0,6),(1,6),(2,4),(3,4),(4,2),(23,2)]:
        contract.set_epoch(model, epoch)
        assert model.transformer.encoder.detail_full_layers == depth
    for step, epoch, multiplier in [(0,0,.1),(500,0,1.),(500,16,.1),(500,22,.01)]:
        contract.update_lr(optimizer,args,epoch,step)
        for group in optimizer.param_groups:
            assert group['lr'] == pytest.approx(group['precision24_peak_lr'] * multiplier)
    saved = dict(epoch=23, precision24_state=contract.state(model,500),
                 criterion_progress=dict(successful_updates=500))
    contract.set_epoch(model,0); contract.restore(model,saved)
    assert model.transformer.encoder.detail_full_layers == 2
    broken = copy.deepcopy(saved); broken['precision24_state']['detail_full_layers'] = 6
    with pytest.raises(ValueError): contract.restore(model,broken)
    with pytest.raises(ValueError): contract.restore(model,dict(epoch=23))


def test_full_optimizer_resume_and_cross_variant_rejection(tmp_path):
    from util.checkpoint import load_native_resume, capture_rng_state
    from util.experiment import variant_signature
    args = recipe()
    model = nn.Module(); model.hidden_dim = 16
    model.backbone = nn.Linear(3,16)
    model.transformer = nn.Module(); model.transformer.encoder = nn.Module()
    contract.configure(model,args); contract.set_epoch(model,2)
    optimizer = torch.optim.AdamW(contract.param_groups(args,model))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[16,22],gamma=.1)
    contract.update_lr(optimizer,args,2,12)
    sum(p.square().mean() for p in model.parameters()).backward(); optimizer.step()
    checkpoint = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        lr_scheduler=scheduler.state_dict(),epoch=2,args=args,
        variant_signature=variant_signature(args),criterion_progress=dict(successful_updates=12),
        precision24_state=contract.state(model,12),rng_states=[capture_rng_state()],
        precision24_parameter_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
        run_metadata=dict(epoch_complete=True,smoke_test=False))
    path=tmp_path/'resume.pth'; torch.save(checkpoint,path)
    duplicate=copy.deepcopy(model)
    new_optimizer=torch.optim.AdamW(contract.param_groups(args,duplicate))
    new_scheduler=torch.optim.lr_scheduler.MultiStepLR(new_optimizer,[16,22],gamma=.1)
    contract.set_epoch(duplicate,0)
    assert load_native_resume(duplicate,path,optimizer=new_optimizer,scheduler=new_scheduler,expected_args=args)==3
    assert duplicate.transformer.encoder.detail_full_layers==4 and args.quality_successful_updates==12
    for a,b in zip(model.parameters(),duplicate.parameters()): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert new_optimizer.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
    wrong=copy.copy(args); wrong.local_refiner_enabled=False
    with pytest.raises(ValueError,match='signature'):
        load_native_resume(duplicate,path,optimizer=new_optimizer,expected_args=wrong)
    broken=copy.deepcopy(checkpoint); broken['precision24_parameter_groups'][0]['precision24_names']=[]
    torch.save(broken,tmp_path/'broken.pth')
    with pytest.raises(ValueError,match='optimizer group'):
        load_native_resume(duplicate,tmp_path/'broken.pth',optimizer=new_optimizer,expected_args=args)


def test_warmstart_allows_only_new_parameters(tmp_path):
    from util.incremental_checkpoint import validate_warmstart
    model=nn.Module(); model.hidden_dim=16; model.backbone=nn.Linear(3,16)
    model.transformer=nn.Module(); model.transformer.encoder=nn.Module()
    saved=copy.deepcopy(model.state_dict())
    args=argparse.Namespace(precision24_enabled=True,detail_context_encoder=True,local_refiner_enabled=True,
                            pretrain_model_path=str(tmp_path/'source.pth'))
    contract.configure(model,args)
    checkpoint=dict(model=saved,run_metadata=dict(epoch_complete=True))
    torch.save(checkpoint,args.pretrain_model_path); validate_warmstart(model,args)
    checkpoint['model'].pop('backbone.weight')
    torch.save(checkpoint,args.pretrain_model_path)
    with pytest.raises(ValueError,match='coverage'): validate_warmstart(model,args)


def test_encoder_separate_query_value_and_p2_gradient():
    from models.aqfcdetr.transformer import DeformableTransformerEncoderLayer
    class Attention(nn.Module):
        def forward(self, q, ref, value, shapes, start, mask):
            assert q.shape[1] == ref.shape[1] == 3 and value.shape[1] == 8
            return q + value.mean(1,keepdim=True)
    layer = DeformableTransformerEncoderLayer(16,32,dropout=0.,n_levels=5,n_heads=4)
    layer.self_attn = Attention()
    memory = torch.randn(2,8,16,requires_grad=True)
    out = layer.forward_queries(memory[:,5:], torch.zeros(2,3,16),
                                torch.zeros(2,3,5,2), memory, None, None)
    (out * torch.randn_like(out)).sum().backward()
    assert memory.grad[:,:5].abs().sum() > 0


@pytest.mark.parametrize('amp', [False, True])
@pytest.mark.parametrize('variant', ['f1_24e', 'f2_24e'])
def test_full_model_identity_dn_and_final_stage_backward(amp, tmp_path, variant):
    if not torch.cuda.is_available(): pytest.skip('native CUDA attention integration')
    import torchvision
    from main import build_model_main
    from util.checkpoint import load_native_resume
    from models.aqfcdetr.query_allocator import QueryBudgetLoss
    args = recipe(variant)
    args.device = 'cuda'; args.distributed = False
    original = torchvision.models.resnet50
    def no_download(*a, **kw):
        kw['weights'] = None
        return original(*a, **kw)
    torch.manual_seed(42)
    with patch('torchvision.models.resnet50', side_effect=no_download):
        model, criterion, post = build_model_main(args)
    model.cuda().eval(); criterion.cuda()
    images = [torch.rand(3,97,129,device='cuda'), torch.rand(3,128,160,device='cuda')]
    with torch.no_grad(), torch.amp.autocast('cuda',enabled=amp):
        reference = model(images)
        refiner = model.local_refiner; fusion = getattr(model.transformer.encoder, 'detail_fusion', None)
        del model.local_refiner
        if fusion is not None: del model.transformer.encoder.detail_fusion
        plain = model(images)
        model.local_refiner = refiner
        if fusion is not None: model.transformer.encoder.detail_fusion = fusion
    for key in ('pred_logits','pred_boxes'):
        maximum = (reference[key].float()-plain[key].float()).abs().max().item()
        print(f'identity AMP={amp} {key} max_abs={maximum}')
        torch.testing.assert_close(reference[key].float(),plain[key].float(),atol=1e-5,rtol=1e-4)
    targets = [dict(boxes=torch.tensor([[.5,.5,.04,.05]],device='cuda'),
                    labels=torch.tensor([0],device='cuda'),size=torch.tensor([97,129],device='cuda')),
               dict(boxes=torch.empty(0,4,device='cuda'),labels=torch.empty(0,dtype=torch.long,device='cuda'),
                    size=torch.tensor([128,160],device='cuda'))]
    optimizer = torch.optim.AdamW(contract.param_groups(args,model),weight_decay=args.weight_decay)
    model.train(); model.set_epoch(23); criterion.train(); criterion.quality_lambda=.25
    budget = QueryBudgetLoss(density_target_backend='vectorized').cuda()
    for epoch in (0,2,4):
        contract.set_epoch(model,epoch)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda',enabled=amp):
            output = model(images,targets)
            assert output['pred_boxes'].shape == output['pred_boxes_coarse'].shape
            assert output['dn_meta']['pad_size'] > 0
            losses = criterion(output,targets)
            loss = sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
            loss = loss + budget(output['allocator_outputs'],dict(targets=targets,
                                 real_counts=torch.tensor([1.,0.],device='cuda')))['loss_allocator_total']
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert model.local_refiner.head[-1].weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1); optimizer.step()
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda',enabled=amp):
        before = model(images)
    path = tmp_path / 'precision_synthetic.pth'
    torch.save(dict(model=model.state_dict(),epoch=4,precision24_state=contract.state(model,3),
                    criterion_progress=dict(successful_updates=3)),path)
    contract.set_epoch(model,0)
    load_native_resume(model,path)
    assert model.transformer.encoder.detail_full_layers == (6 if variant == 'f2_24e' else 2)
    with torch.no_grad(), torch.amp.autocast('cuda',enabled=amp):
        after = model(images)
    torch.testing.assert_close(before['pred_boxes'],after['pred_boxes'],atol=1e-5,rtol=1e-4)
    predictions = post['bbox'](after,torch.tensor([[97,129],[128,160]],device='cuda'))
    assert all(torch.isfinite(p['boxes']).all() for p in predictions)
    # Reverse group order deliberately; nonzero refinement must follow restored query order.
    def budgets(_module, _inputs, result):
        result['query_counts'] = torch.tensor([500,300],device='cuda')
        return result
    model.transformer.eval_query_floor = 0
    model.transformer.force_query_budget = None
    handle = model.transformer.query_allocator.register_forward_hook(budgets)
    grouped_states = []
    refiner_queries = []
    decoder_hook = model.transformer.decoder.register_forward_hook(
        lambda _m, _i, result: grouped_states.append(result[0][-1].detach().clone()))
    refiner_hook = model.local_refiner.register_forward_pre_hook(
        lambda _m, inputs: refiner_queries.append(inputs[0].detach().clone()))
    try:
        with torch.no_grad(), torch.amp.autocast('cuda',enabled=amp):
            model.transformer.grouped_decoder_inference = True
            grouped = model(images)
            decoder_hook.remove(); refiner_hook.remove()
            model.transformer.grouped_decoder_inference = False
            padded = model(images)
        assert grouped['query_valid_mask'].sum(1).tolist() == [500,300]
        assert len(grouped_states) == 2 and len(refiner_queries) == 1
        # This checks the actual scatter/slice contract exactly, even with AMP kernels.
        assert torch.equal(refiner_queries[0][1,:300],grouped_states[0][0])
        assert torch.equal(refiner_queries[0][0,:500],grouped_states[1][0])
        valid = grouped['query_valid_mask']
        for key in ('pred_logits','pred_boxes_coarse','pred_boxes'):
            error = (grouped[key][valid].float()-padded[key][valid].float()).abs().max().item()
            print(f'grouped AMP={amp} {key} max_abs={error}')
            if not amp:
                torch.testing.assert_close(grouped[key][valid].float(),padded[key][valid].float(),atol=1e-5,rtol=1e-4)
            # Different half-precision batch/sequence GEMM shapes need not be numerically
            # identical. Report the error, do not widen the FP32 bound or use it for index checks.
            assert torch.isfinite(grouped[key][valid]).all()
    finally:
        handle.remove()
        decoder_hook.remove(); refiner_hook.remove()
