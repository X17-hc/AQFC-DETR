"""H1 from the historical detector: new 24 epochs, not S2/F2 continuation."""
_base_ = ['../incremental_v2/p2_24e_legacy.py']
architecture_variant = 'legacy_joint_v2'
precision24_enabled = False
train_transform_mode = 'native_multiscale'
data_aug_scales = [640, 704, 768, 800]
data_aug_max_size = 1333
joint_warmup_updates = 500
joint_new_lr = 1e-4
joint_subset_epochs = [0, 3, 7, 12]
joint_subset_ids = '/workspace/AQFC-DETR/outputs/legacy24_diagnosis_20260917/subset_ids.json'
epochs = 24
val_epoch = [23]
save_checkpoint_interval = 1
geometry_loss_weight = 0.0
batch_size = 2
use_ema = False
aligned_box_loss = True
batched_metric_transfer = True
non_blocking_transfer = True

