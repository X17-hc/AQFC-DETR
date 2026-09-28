"""Same model and 24-epoch clocks. Step limits are explicit runtime smoke flags."""
_base_ = ['h1_24e.py']
val_epoch = [0]
joint_subset_epochs = []

