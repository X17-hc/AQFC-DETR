_base_ = ['aqfc_r50_5scale_local8gb.py']
epochs = 1
val_epoch = [0]
mosaic_p = 0.0
copy_paste_p = 0.0
data_aug_scales = [256]
data_aug_max_size = 320
allocator_schedule = dict(warmup_epochs=0, peak_weight=1.0, final_weight=1.0)
