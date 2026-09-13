# Research: launch with --no-pretrained; do not initialize from old detector weights.
_base_ = ['../../aitodv2/aqfc_r50_5scale_local8gb.py']
allocator_encoder_type = 'standard'
allocator_enabled = True
calibrator_enabled = True
density_target_chunk_size = 512
spatial_semantic_ratio = 0.75
spatial_grid_size = [8, 8]
eval_backend = 'legacy'
run_purpose = 'research'
train_split = 'train'
eval_split = 'val'
val_epoch = list(range(24))
density_target_backend = 'vectorized'
