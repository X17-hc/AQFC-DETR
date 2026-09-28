"""Dedicated trainer: two independent adapter epochs, never detector fine-tuning."""
_base_ = ['./reference.py']
epochs = 2
lr = 1e-4
weight_decay = 1e-4
clip_max_norm = 0.1
val_epoch = []
