"""Regress actual Transformer training selection before Decoder computation."""
from unittest.mock import patch
import argparse
import pytest
import torch
import torchvision
from main import build_model_main
from util.slconfig import SLConfig
from models.aqfcdetr.proposal_selection import select_proposal_indices


@pytest.mark.parametrize('mode',['semantic','fused','mixed','spatial'])
@pytest.mark.parametrize('tied',[False,True])
def test_budget_selection_matches_individual_calls_and_gathers(mode,tied):
    torch.manual_seed(42)
    logits=torch.randn(4,64,8) if not tied else torch.zeros(4,64,8)
    density=torch.rand(4,64)
    padding=torch.zeros(4,64,dtype=torch.bool)
    padding[0,56:]=True
    proposals=torch.rand(4,64,4)
    proposals[1,48:]=float('inf')
    counts=torch.tensor([8,32,16,24])
    kw=dict(mode=mode,spatial_shapes=torch.tensor([[8,8]]),spatial_grid_size=(2,4))
    reports=[]
    result=select_proposal_indices(logits,density,padding,32,proposal_boxes=proposals,
                                   query_counts=counts,diagnostics=reports,**kw)
    assert result.shape==(4,32)
    memory=torch.randn(4,64,256,requires_grad=True)
    gathered=memory.gather(1,result[:,:,None].expand(-1,-1,256))
    mask=torch.arange(32)[None]<counts[:,None]
    for b,k in enumerate(counts.tolist()):
        expected=select_proposal_indices(logits[b:b+1],density[b:b+1],padding[b:b+1],k,
                                         proposal_boxes=proposals[b:b+1],**kw)[0]
        assert torch.equal(result[b,:k],expected)
        assert result[b,:k].unique().numel()==k
        assert not padding[b,result[b,:k]].any()
        assert torch.isfinite(proposals[b,result[b,:k]]).all()
        torch.testing.assert_close(gathered[b,:k],memory[b,expected],atol=0,rtol=0)
    gathered[mask].sum().backward()
    for b,k in enumerate(counts.tolist()):
        assert torch.equal(memory.grad[b,:,0].nonzero().flatten().sort().values,result[b,:k].sort().values)
        assert (memory.grad[b,result[b,:k]]==1).all()
    if mode=='spatial':
        assert len(reports)==4
        assert [r['semantic_reserved']+r['spatial_added']+r['filled'] for r in reports]==counts.tolist()
    # No change for equal budgets, including ties and all-zero density fallback.
    density.zero_()
    old=select_proposal_indices(logits,density,padding,16,proposal_boxes=proposals,**kw)
    new=select_proposal_indices(logits,density,padding,16,proposal_boxes=proposals,
                               query_counts=torch.full((4,),16),**kw)
    assert torch.equal(old,new)


@pytest.mark.parametrize('mode',['spatial','mixed'])
def test_real_training_uses_each_images_budget(mode):
    if not torch.cuda.is_available():
        pytest.skip('Native CUDA synthetic integration')
    torch.manual_seed(42)
    config=SLConfig.fromfile('configs/experiments/local8gb/light_spatial.py')._cfg_dict.to_dict()
    config.update(device='cuda',distributed=False,use_dn=True,dn_number=4)
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,criterion,post=build_model_main(argparse.Namespace(**config))
    model.cuda().train();criterion.cuda()
    transformer=model.transformer
    transformer.proposal_selection_mode=mode
    transformer.spatial_grid_size=(2,2)
    transformer.query_budget_levels=[8,16]
    transformer.force_query_budget=None
    original_allocator=transformer.query_allocator.forward
    def allocate(*a,**kw):
        result=original_allocator(*a,**kw)
        result['query_counts']=torch.tensor([8,16],device='cuda')
        return result
    original_select=transformer.select_proposal_indices
    seen=[]
    def check(logits,density,padding,topk,**kw):
        actual=original_select(logits,density,padding,topk,**kw)
        boxes=kw['proposal_boxes']
        for b,k in enumerate([8,16]):
            expected=select_proposal_indices(logits[b:b+1],density[b:b+1],padding[b:b+1],k,
                mode=mode,density_weight=transformer.proposal_density_weight,
                mixed_density_ratio=transformer.mixed_density_ratio,proposal_boxes=boxes[b:b+1],
                spatial_shapes=kw['spatial_shapes'],spatial_semantic_ratio=transformer.spatial_semantic_ratio,
                spatial_grid_size=transformer.spatial_grid_size)
            assert torch.equal(actual[b,:k],expected[0]), 'Training selection differs from direct per-image budget'
        seen.append(actual)
        return actual
    images=[torch.rand(3,128,128,device='cuda') for _ in range(2)]
    targets=[dict(boxes=torch.tensor([[.5,.5,.1,.1]],device='cuda'),
                  labels=torch.tensor([0],device='cuda')) for _ in range(2)]
    with patch.object(transformer.query_allocator,'forward',side_effect=allocate), \
         patch.object(transformer,'select_proposal_indices',side_effect=check):
        with torch.amp.autocast('cuda'):
            output=model(images,targets)
            losses=criterion(output,targets)
            loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
    assert len(seen)==1  # One batched training Decoder, including DN.
    assert output['query_valid_mask'].sum(1).tolist()==[8,16]
    assert output['dn_meta']['pad_size']>0
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
    torch.optim.AdamW(model.parameters(),lr=1e-5).step()
    assert len(post['bbox'](output,torch.tensor([[128,128],[128,128]],device='cuda')))==2
    # Both inference execution modes must now select identical real proposals
    # for the same encoder batch; do not compare different backbone batches.
    model.eval()
    with patch.object(transformer.query_allocator,'forward',side_effect=allocate),torch.no_grad():
        transformer.grouped_decoder_inference=True
        grouped=model(images)
        transformer.grouped_decoder_inference=False
        padded=model(images)
    valid=padded['query_valid_mask']
    assert torch.equal(valid,grouped['query_valid_mask'])
    torch.testing.assert_close(grouped['interm_outputs_for_matching_pre']['pred_boxes'][valid],
                               padded['interm_outputs_for_matching_pre']['pred_boxes'][valid],atol=0,rtol=0)
    torch.testing.assert_close(grouped['pred_boxes'][valid],padded['pred_boxes'][valid],atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(grouped['pred_logits'][valid],padded['pred_logits'][valid],atol=1e-5,rtol=1e-4)


def test_old_candidate_contract_cannot_silently_resume(tmp_path):
    from util.experiment import variant_signature
    from util.checkpoint import load_native_resume
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
    args=argparse.Namespace(classification_loss_type='quality_blend',training_phase_epoch_offset=11,
                            training_phase_total_epochs=24)
    signature=variant_signature(args)
    signature['correctness_revision']='incremental_v2'
    path=tmp_path/'old_contract.pth'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=0,
                    variant_signature=signature,criterion_progress={'successful_updates':1}),path)
    with pytest.raises(ValueError,match='signature'):
        load_native_resume(model,path,optimizer=optimizer,expected_args=args)


def test_real_budget_levels_and_clipped_capacity():
    torch.manual_seed(7)
    logits=torch.randn(4,1600,9);density=torch.rand(4,1600)
    padding=torch.zeros(4,1600,dtype=torch.bool)
    counts=torch.tensor([300,500,900,1500])
    kw=dict(mode='spatial',spatial_shapes=torch.tensor([[40,40]]))
    result=select_proposal_indices(logits,density,padding,1500,query_counts=counts,**kw)
    for b,k in enumerate(counts.tolist()):
        expected=select_proposal_indices(logits[b:b+1],density[b:b+1],padding[b:b+1],k,**kw)
        assert torch.equal(result[b,:k],expected[0])
    # Budgets already clipped by Transformer to legal proposal capacities.
    boxes=torch.zeros(2,8,4);boxes[0,1:]=float('inf')
    padding=torch.zeros(2,8,dtype=torch.bool);padding[1,6:]=True
    logits=torch.randn(2,8,9);density=torch.zeros(2,8)
    result=select_proposal_indices(logits,density,padding,6,proposal_boxes=boxes,
        query_counts=torch.tensor([1,6]),mode='spatial',spatial_shapes=torch.tensor([[2,4]]))
    assert result[0].tolist()==[0]*6
    assert result[1].unique().numel()==6 and (result[1]<6).all()
