"""C0: final epoch23 ordinary model -> three fresh, low-LR engineering epochs.

Not native resume: optimizer/scheduler are new. Do not replace the frozen source
with C1/C0 output. Test data is for engineering diagnosis, not model selection.
"""
_base_ = ['../incremental_v2/p2_24e_legacy.py']
epochs = 3
val_epoch = [2]
save_checkpoint_interval = 1
lr = 1e-5
lr_backbone = 1e-6
multi_step_lr = False
onecyclelr = False
lr_drop = 100  # No decay during these three epochs.
lr_drop_list = []
use_ema = False
batch_size = 2
train_split = 'trainval'
eval_split = 'test'
run_purpose = 'engineering_check'

# Repeat the TERMINAL phase of the old recipe, not the teacher/mosaic warmup.
# Local epochs 0..2 count new optimizer work; phase epoch remains 23/24.
training_phase_epoch_offset = 0
training_phase_total_epochs = 24
training_phase_fixed_epoch = 23
classification_loss_type = 'quality_blend'
quality_blend_max = 0.25
quality_blend_warmup_epochs = 0  # Already trained: lambda stays 0.25 in BOTH arms.

strict_warmstart = True
expected_pretrained_epoch = 23
expected_pretrained_sha256 = 'c447367e2d19411319a990985e0127af8db08bfb7eec0329b2065b095ccc64eb'

geometry_loss_weight = 0.0
geometry_min_size_pixels = 4.0
geometry_smooth_l1_beta = 0.1
geometry_warmup_epochs = 1
