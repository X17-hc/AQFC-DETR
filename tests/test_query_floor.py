from types import SimpleNamespace
import pytest
import torch
from util.runtime import apply_eval_query_floor, validate_eval_query_floor, update_skipped_streak


def test_floor_preserves_high_budgets():
    original = torch.tensor([300, 500, 900, 1500])
    assert apply_eval_query_floor(original) is original
    assert apply_eval_query_floor(original, 900).tolist() == [900, 900, 900, 1500]
    assert original.tolist() == [300, 500, 900, 1500]
    assert torch.minimum(apply_eval_query_floor(original, 900), torch.tensor([200, 1000, 1000, 1200])).tolist() == [200,900,900,1200]
    with pytest.raises(ValueError):
        apply_eval_query_floor(original, 900, training=True)


@pytest.mark.parametrize('floor,evaluation,forced', [(900,False,None),(900,True,900),(800,True,None),(-1,True,None)])
def test_invalid_floor(floor,evaluation,forced):
    with pytest.raises(ValueError):
        validate_eval_query_floor(floor,evaluation,[300,500,900,1500],forced)


def test_valid_floor_and_skips():
    validate_eval_query_floor(900,True,[300,500,900,1500])
    validate_eval_query_floor(0,False,[300,500,900,1500],900)
    assert update_skipped_streak(19,True,20) == 0
    assert update_skipped_streak(18,False,20) == 19
    with pytest.raises(FloatingPointError):
        update_skipped_streak(19,False,20)
    assert update_skipped_streak(19,False,0) == 20


def test_queue_training_and_eval_are_separate(tmp_path):
    from tools.continue_diagnosis_epoch import cmd, UPDATED
    train = cmd(tmp_path,'train',UPDATED,train=True)
    assert '--eval' not in train and '--eval-query-floor' not in train
    assert 'epochs=2' in train and 'val_epoch=[]' in train
    assert '--pretrained' not in train and '--resume' in train
    evaluation = cmd(tmp_path,'eval',UPDATED,ids='subset_ids.json',floor=900)
    assert '--eval' in evaluation and evaluation[evaluation.index('--eval-query-floor')+1]=='900'
