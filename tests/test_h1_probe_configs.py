"""Probe recipes lock H1 epoch23. No torch or addict import."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H1_EPOCH23_SHA = '15862ce9658c871c0152c5120f1562675a5ce294bf7029a4b7ed28b4bbbd9aee'
PROBES = ('h1_finetune_3e', 'h1_ema_3e', 'h1_joint_dfl_035_3e', 'h1_glu_3e')


def recipe(name):
    chain, current = [], name
    while current:
        ns = {}
        path = ROOT / 'configs/legacy_joint' / f'{current}.py'
        exec(path.read_text(encoding='utf-8'), ns)
        chain.append(ns)
        bases = ns.get('_base_') or []
        base = bases[0] if isinstance(bases, list) and bases else None
        if not base or '/' in base or base.startswith('.'):
            break
        current = Path(base).stem
    values = {}
    for ns in reversed(chain):
        values.update({k: v for k, v in ns.items() if not k.startswith('_')})
    return values


def test_probes_lock_h1_epoch23_and_keep_geometry_off():
    for name in PROBES:
        values = recipe(name)
        assert values['joint_finetune_from_h1'] is True
        assert values['epochs'] == 3
        assert values['val_epoch'] == [2]
        assert values['expected_pretrained_sha256'] == H1_EPOCH23_SHA
        assert values['geometry_loss_weight'] == 0
        assert values['lr'] == 1e-5
        assert values['lr_backbone'] == 1e-6
        assert values['joint_subset_epochs'] == []


def test_single_knob_probes():
    ema = recipe('h1_ema_3e')
    assert ema['use_ema'] is True and ema['joint_allow_ema'] is True
    dfl = recipe('h1_joint_dfl_035_3e')
    assert dfl['joint_dfl_coef'] == 0.35 and dfl['joint_o2m_coef'] == 0.25
    assert dfl['use_ema'] is False
    glu = recipe('h1_glu_3e')
    assert glu['calibrator_gate_type'] == 'glu'
    assert glu['use_ema'] is False
    assert recipe('h1_finetune_3e').get('calibrator_gate_type', 'tanh') != 'swish'
