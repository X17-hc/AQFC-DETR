"""P2: 24 NEW fine-tuning epochs from the frozen epoch10 model (not resume).

Use the dedicated PyCharm launcher. Keep the original three-epoch P2 untouched.
This is a longer engineering experiment, not a same-workload P1/P2 comparison.
"""
_base_ = ['p2.py']

# Local epochs 0..23 correspond to model/augmentation phases 11..34.
# Extend the phase horizon explicitly: never restart the allocator teacher.
epochs = 24
training_phase_epoch_offset = 11
training_phase_total_epochs = 35
val_epoch = [23]  # One full test evaluation after the final training epoch.
save_checkpoint_interval = 1

# Model-only warm-start; a new optimizer and a new 24-epoch LR schedule.
# New epochs 1..16: 1e-5; 17..22: 1e-6; 23..24: 1e-7 (backbone /10).
lr = 1e-5
lr_backbone = 1e-6
onecyclelr = False
multi_step_lr = True
lr_drop_list = [16, 22]
lr_drop = 100  # Not used for LR with MultiStepLR; kept for checkpoint logic.

# Match P2: ordinary model, decoder matching-query quality blend only.
use_ema = False
classification_loss_type = 'quality_blend'
quality_blend_max = 0.25
quality_blend_warmup_epochs = 1

# Freeze the same initialization as the short P2 experiment. No silent fallback.
strict_warmstart = True
expected_pretrained_epoch = 10
expected_pretrained_sha256 = '898f7542b4c7b96317e857d9df5708ed1de63a8644f9edfaed3f080b2e7e272a'
