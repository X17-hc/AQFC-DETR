from pathlib import Path
import pytest

from main import PROJECT_ROOT, get_args_parser, resolve_launch_defaults


def parse(argv):
    return resolve_launch_defaults(get_args_parser().parse_args(argv))


def test_no_arguments_use_local_training_defaults():
    args = parse([])
    assert Path(args.config_file) == PROJECT_ROOT / 'configs/aitodv2/aqfc_r50_5scale_local8gb.py'
    assert args.coco_path == 'D:/PythonProject/DQ-DETR/DQ-DETR/data/path/AITODv2'
    assert Path(args.pretrain_model_path) == PROJECT_ROOT / 'weights/legacy/dqdetr_best305.pth'
    assert Path(args.output_dir) == PROJECT_ROOT / 'outputs/aitodv2_pycharm'
    assert args.num_workers == 0
    assert not args.eval and args.max_train_steps == 0


def test_cli_overrides_all_local_settings():
    args = parse(['--config','custom.py','--data-root','other/data',
                  '--pretrained','other.pth','--output-dir','other/output','--num_workers','2'])
    assert (args.config_file,args.coco_path,args.pretrain_model_path,args.output_dir,args.num_workers) == (
        'custom.py','other/data','other.pth','other/output',2)


def test_resume_does_not_enable_default_pretrained():
    assert parse(['--resume','checkpoint.pth']).pretrain_model_path == ''
    assert parse(['--resume','checkpoint.pth','--eval']).pretrain_model_path == ''


@pytest.mark.parametrize('weight', ['explicit.pth', str(PROJECT_ROOT / 'weights/legacy/dqdetr_best305.pth')])
@pytest.mark.parametrize('reverse', [False, True])
def test_explicit_pretrained_conflicts_with_resume(weight, reverse):
    argv = ['--resume','checkpoint.pth','--pretrained',weight]
    if reverse:
        argv = argv[2:] + argv[:2]
    with pytest.raises(ValueError, match='only one'):
        parse(argv)


def test_pretrained_default_is_declared_in_argument_parser():
    args = get_args_parser().parse_args([])
    assert Path(args.pretrain_model_path) == PROJECT_ROOT / 'weights/legacy/dqdetr_best305.pth'


@pytest.mark.parametrize('argv', [
    ['--pretrained','explicit.pth','--no-pretrained'],
    ['--no-pretrained','--pretrained','explicit.pth'],
])
def test_explicit_pretrained_cannot_be_silently_disabled(argv):
    with pytest.raises(ValueError, match='cannot be used together'):
        parse(argv)


def test_resume_with_no_pretrained_is_allowed():
    assert parse(['--resume','checkpoint.pth','--no-pretrained']).pretrain_model_path == ''


def test_default_pretrained_can_be_disabled():
    assert parse(['--no-pretrained']).pretrain_model_path == ''


def test_default_project_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse([])
    assert Path(args.config_file).parent.parent.parent == PROJECT_ROOT
    assert Path(args.pretrain_model_path).parent.parent.parent == PROJECT_ROOT
    assert Path(args.output_dir).parent.parent == PROJECT_ROOT
