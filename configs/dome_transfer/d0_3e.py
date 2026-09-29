"""Mature H1 control; NEW optimizer, same ordinary epoch23 initialization as D1."""
_base_ = ['../legacy_joint/h1_24e.py']
dome_transfer_recipe = 'd0'
density_underestimate_weight = 0.0
density_underestimate_warmup_updates = 500
protected_density_valid_classes = list(range(8))
epochs = 3
val_epoch = [2]
joint_subset_epochs = []
joint_warmup_updates = 500
joint_new_lr = 1e-5
lr = 1e-5
lr_backbone = 1e-6
lr_linear_proj_mult = 0.1
lr_drop_list = []
training_phase_fixed_epoch = 23
training_phase_epoch_offset = 0
training_phase_total_epochs = 24
quality_blend_warmup_epochs = 0
quality_blend_max = 0.25
expected_pretrained_epoch = 23
expected_pretrained_sha256 = '15862ce9658c871c0152c5120f1562675a5ce294bf7029a4b7ed28b4bbbd9aee'
proposal_selection_mode = 'spatial'

