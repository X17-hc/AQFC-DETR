"""Raw VisDrone IDs 1..10, train/val, ImageNet-only initialization."""
_base_ = ['../visdrone/aqfc_r50_5scale_24e.py']
train_split = 'train'
eval_split = 'val'
allocator_encoder_type = 'standard'
density_target_backend = 'reference'
run_purpose = 'research'
eval_backend = 'legacy'
calibrator_gate_type = 'tanh'
batch_size = 2
amp_init_scale = 32.0
