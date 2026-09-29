"""Partial engineering check only; do not use its checkpoint to start formal runs."""
_base_ = ['d1_3e.py']
dome_transfer_smoke = True
epochs = 1
val_epoch = [0]
