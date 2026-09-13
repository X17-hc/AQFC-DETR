"""A6000 paper-style scales/optimizer, but AQFC (not an exact paper reproduction)."""
_base_ = ['../aitodv2/aqfc_r50_5scale_24e.py']
allocator_encoder_type = 'standard'
density_target_backend = 'reference'
allocator_enabled = True
calibrator_enabled = True
proposal_selection_mode = 'fused'
calibrator_gate_type = 'tanh'
calibrator_use_spatial = True
run_purpose = 'engineering_check'
eval_backend = 'legacy'
train_split = 'trainval'
eval_split = 'test'
batch_size = 2
epochs = 24
save_checkpoint_interval = 1
val_epoch = [23]
amp_init_scale = 32.0
