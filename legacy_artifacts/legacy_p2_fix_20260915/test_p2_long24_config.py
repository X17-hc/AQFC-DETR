"""No GPU/data/model execution: verify the long-run launch contract."""
import argparse
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import pytest
from util.slconfig import SLConfig
from util.config_validation import validate_config
from util.incremental import training_phase, quality_progress

ROOT = Path(__file__).resolve().parents[1]


def config():
    values = SLConfig.fromfile(str(ROOT / 'configs/incremental_v2/p2_24e.py'))._cfg_dict.to_dict()
    validate_config(values)
    return values


def test_long24_phase_and_recipe():
    c = config()
    args = argparse.Namespace(**c)
    assert training_phase(args, 0) == (11, 35)
    assert training_phase(args, 23) == (34, 35)
    assert c['epochs'] == 24 and c['val_epoch'] == [23]
    assert c['save_checkpoint_interval'] == 1 and not c['use_ema']
    assert c['train_split'] == 'trainval' and c['eval_split'] == 'test'
    assert c['run_purpose'] == 'engineering_check'
    assert c['batch_size'] == 2 and c['classification_loss_type'] == 'quality_blend'
    assert all(c[key] for key in ('aligned_box_loss', 'batched_metric_transfer', 'non_blocking_transfer'))
    assert c['strict_warmstart'] and c['expected_pretrained_epoch'] == 10
    assert quality_progress(args, 0, 7009) == 0
    assert quality_progress(args, 7009, 7009) == .25
    old = SLConfig.fromfile(str(ROOT / 'configs/incremental_v2/p2.py'))
    assert old.epochs == 3 and old.training_phase_total_epochs == 24


def test_long24_actual_lr_schedule():
    import torch
    c = config()
    p = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([p], lr=c['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=c['lr_drop_list'])
    rates = []
    for _ in range(24):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx([1e-5] * 16 + [1e-6] * 6 + [1e-7] * 2)


def test_long24_pycharm_real_parser():
    from main import get_args_parser
    doc = ET.parse(ROOT / '.run/AQFC-DETR_P2_24e.run.xml')
    opts = {x.get('name'): x.get('value') for x in doc.findall('.//option')}
    assert opts['WORKING_DIRECTORY'] == '/workspace/AQFC-DETR'
    assert opts['SDK_HOME'] == '/opt/conda/envs/AQFC-DETR/bin/python'
    assert opts['SCRIPT_NAME'] == '$PROJECT_DIR$/main.py'
    assert doc.find('.//env[@name="CUDA_VISIBLE_DEVICES"]').get('value') == '3'
    args = get_args_parser().parse_args(shlex.split(opts['PARAMETERS'].replace('/workspace/AQFC-DETR', ROOT.as_posix())))
    assert args.unique_output_dir and not args.resume and not args.eval
    assert args.num_workers == 2 and args.amp and args.seed == 42
    assert args.max_train_steps == args.max_eval_steps == args.start_epoch == 0
    assert args.pretrain_model_path.endswith('/20260910_235559_501337/checkpoint0010.pth')
    assert not {k for k in config() if k in vars(args) and vars(args)[k] is not None}
