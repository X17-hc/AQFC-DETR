import torch
from models.aqfcdetr.dn_components import prepare_for_cdn


def test_zero_dn_number_generates_no_queries():
    targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5,.5,.1,.1]]))]
    result = prepare_for_cdn((targets, 0, .5, 1.), True, 300, 2, 256, torch.nn.Embedding(2,256))
    assert result == (None, None, None, None)


def test_all_empty_targets_generate_no_dn_queries():
    targets = [dict(labels=torch.empty(0,dtype=torch.long), boxes=torch.empty(0,4))]
    result = prepare_for_cdn((targets, 100, .5, 1.), True, 300, 2, 256, torch.nn.Embedding(2,256))
    assert result == (None, None, None, None)


def test_dn_uses_label_embedding_device():
    targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5,.5,.1,.1]]))]
    labels, boxes, mask, metadata = prepare_for_cdn(
        (targets, 2, .5, 1.), True, 300, 2, 256, torch.nn.Embedding(2,256))
    assert labels.device.type == boxes.device.type == mask.device.type == 'cpu'
    assert labels.shape[1] == metadata['pad_size']
