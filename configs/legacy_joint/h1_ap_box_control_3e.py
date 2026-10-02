"""3-epoch box-head control from the scheme-2 epoch23 weights.

Same freeze, learning rate, and epoch count as the tighten run. The box loss
stays the original L1 / GIoU / NWD.
"""
_base_ = ['h1_aqfc_refactor_24e.py']
native_weight_finetune = True
box_head_only = True
box_tighten = False
epochs = 3
val_epoch = []
joint_subset_epochs = [2]
joint_warmup_updates = 1
lr = 1e-5
lr_backbone = 1e-5
joint_new_lr = 1e-5
multi_step_lr = False
lr_drop_list = []
