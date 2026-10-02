"""1-epoch distribution-refiner control from the scheme-2 epoch23 weights.

The refiner trains with the existing box loss. The log-width loss stays off.
"""
_base_ = ['h1_aqfc_refactor_24e.py']
native_weight_finetune = True
edge_refine_only = True
edge_logwh = False
decoder_heads_only = False
box_head_only = False
box_tighten = False
epochs = 1
val_epoch = []
joint_subset_epochs = []
joint_warmup_updates = 1
lr = 1e-5
lr_backbone = 1e-5
joint_new_lr = 1e-5
multi_step_lr = False
lr_drop_list = []
expected_native_sha256 = '222b7a0dd797d25c2105084bc6fa4fb8374ef01967acfff0d3d1ce0eca2d0313'
