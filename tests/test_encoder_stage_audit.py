"""Focused phase-audit and real deformable-attention query-slice contracts."""
import json
from pathlib import Path
import pytest
import torch

from tools.audit_encoder_stages import select_ids, check_exports, quality_review


def test_quality_collapse_is_not_reported_as_training_acceptance():
    results = {k: {'metrics': {'test_coco_eval_bbox': [v]}} for k,v in
               [('f1_6',.4242),('f1_4',.2328),('f1_2',.0274)]}
    review = quality_review(results)
    assert review['formal_training_acceptance'] == 'NOT_ESTABLISHED'
    assert all(x['review_required'] for x in review['stages'].values())


def test_ids_are_fixed_unique_and_keep_original_smoke():
    annotation = {'images': [{'id': x} for x in reversed(range(100))]}
    ids = select_ids(annotation, 32)
    assert ids[:2] == [0, 1] and len(set(ids)) == 32
    assert ids == select_ids(annotation, 32)
    with pytest.raises(ValueError):
        select_ids({'images': [{'id': 1}, {'id': 1}]}, 2)


def test_export_coverage_fails_on_wrong_image_order(tmp_path):
    file = tmp_path/'images.jsonl'
    file.write_text('\n'.join(json.dumps({'image_id': x}) for x in [1, 2]))
    assert len(check_exports(tmp_path, [1, 2])) == 2
    with pytest.raises(RuntimeError):
        check_exports(tmp_path, [2, 1])


@pytest.mark.parametrize('amp', [False, True])
def test_native_attention_query_slice_equals_full_query(amp):
    if not torch.cuda.is_available():
        pytest.skip('Native CUDA attention required')
    from models.aqfcdetr.transformer import DeformableTransformerEncoderLayer
    torch.manual_seed(42)
    layer = DeformableTransformerEncoderLayer(32,64,dropout=0.,n_levels=5,n_heads=4).cuda().eval()
    # Odd rectangular levels plus padding. Query dimension is independent of the value dimension.
    shapes = torch.tensor([[7,9],[4,5],[2,3],[1,2],[1,1]],device='cuda')
    sizes = shapes.prod(1)
    starts = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
    n = int(sizes.sum()); cut = int(sizes[0])
    memory = torch.randn(2,n,32,device='cuda',requires_grad=True)
    pos = torch.randn_like(memory)
    refs = torch.rand(2,n,5,2,device='cuda')
    mask = torch.zeros(2,n,dtype=torch.bool,device='cuda'); mask[0,8:63:9] = True
    with torch.autocast('cuda',enabled=amp):
        full = layer(memory,pos,refs,shapes,starts,mask)
        partial = layer.forward_queries(memory[:,cut:],pos[:,cut:],refs[:,cut:],memory,shapes,starts,mask)
    # Same AMP dtype/path; do not relax the agreed FP32 tolerance.
    torch.testing.assert_close(partial, full[:,cut:],atol=1e-5,rtol=1e-4)
    (partial.float()*torch.randn_like(partial)).sum().backward()
    assert torch.isfinite(memory.grad).all() and memory.grad[:,:cut].abs().sum() > 0
