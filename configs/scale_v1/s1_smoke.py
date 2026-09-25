"""Manual 20-step/2-batch smoke entry; never resume it for the formal run."""
_base_ = ['./s1_3e.py']
epochs = 1
val_epoch = [0]
