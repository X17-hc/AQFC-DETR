"""Shared 3-epoch low-LR finetune from frozen H1 epoch23. New optimizer."""
_base_ = ['h1_24e.py']
joint_finetune_from_h1 = True
epochs = 3
val_epoch = [2]
joint_subset_epochs = []
joint_warmup_updates = 500
joint_new_lr = 1e-5
lr = 1e-5
lr_backbone = 1e-6
multi_step_lr = False
lr_drop_list = []
use_ema = False
quality_blend_warmup_epochs = 0
training_phase_fixed_epoch = 23
training_phase_epoch_offset = 0
training_phase_total_epochs = 24
expected_pretrained_epoch = 23
expected_pretrained_sha256 = '15862ce9658c871c0152c5120f1562675a5ce294bf7029a4b7ed28b4bbbd9aee'
# Contrast is H1 e23 full test 32.2 / 15.6 and D0 32.2 / 16.0.
# Promotion: +0.3 AP and +0.3 APvt, AP75>=-0.2pp. Do not change H1/D0/D1 weight dirs.
# joint_subset_epochs=[] ; promotion uses 14018 only. geometry_loss_weight stays 0.
# Best-checkpoint stays off on test. Run 32 then 1000 precheck before this 3e.
