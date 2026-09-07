_base_ = ['aqfc_r50_5scale_24e.py']
# Same architecture and query budgets; a smaller batch and input resolution
# make local training feasible on the RTX 4060 Laptop (8 GiB).
batch_size = 1
data_aug_scales = [256, 288, 320]
data_aug_max_size = 384
data_aug_scales2_resize = [256, 288]
data_aug_scales2_crop = [192, 256]
use_ema = False
# Observed stable scale after the September RTX 4060 calibration run.
# GradScaler still adapts dynamically; this is not a substitute for finite-loss checks.
amp_init_scale = 32.0
# Dense images already approach 8 GiB; merging four images can exceed it.
mosaic_p = 0.0
copy_paste_p = 0.0
val_epoch = [0, 5, 11, 17, 23]
