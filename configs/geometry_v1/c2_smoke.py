"""Manual engineering smoke check; never reuse its checkpoint for C2 training.

The run configuration limits this to 20 train steps and 2 evaluation batches.
The normal geometry warmup remains unchanged, so this checks early warmup,
not the full 0.10 plateau and not final accuracy.
"""
_base_ = ['c2_3e.py']
epochs = 1
val_epoch = [0]
