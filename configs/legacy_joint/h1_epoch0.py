"""Same 24-epoch schedule, full evaluation at epoch0; launcher stops after one."""
_base_ = ['h1_24e.py']
val_epoch = [0, 23]

