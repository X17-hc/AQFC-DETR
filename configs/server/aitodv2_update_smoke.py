"""Same batch/scales/EMA/augmentation policy; CLI bounds both loaders."""
_base_ = ['aitodv2_update_24e.py']
epochs = 1
val_epoch = [0]
