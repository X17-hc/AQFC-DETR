# Engineering check only; old detector initialization is not research evidence.
_base_ = ['../../aitodv2/aqfc_r50_5scale_local8gb.py']
allocator_encoder_type = 'standard'
allocator_enabled = True
calibrator_enabled = True
density_target_backend = 'reference'
density_target_chunk_size = 512
spatial_semantic_ratio = 0.75
spatial_grid_size = [8, 8]
eval_backend = 'legacy'
run_purpose = 'engineering_check'
train_split = 'trainval'
eval_split = 'test'
