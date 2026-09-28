"""Independent 20-step smoke; its checkpoint must never resume the formal run."""
_base_ = ['./f2_24e.py']
epochs = 1
val_epoch = [0]
