"""Manual smoke only; formal training starts again from the frozen epoch23."""
_base_ = ['./s2_3e.py']
epochs = 1
val_epoch = [0]
