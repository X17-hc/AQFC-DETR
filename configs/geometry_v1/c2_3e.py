"""C2: bounded geometry-weight sensitivity test, NOT a continuation of C1.

All settings inherit C1; only the plateau weight changes from 0.05 to 0.10.
Start from the frozen epoch23 ordinary model with a fresh optimizer.
Keep the one-epoch successful-update warmup, loss formula and decoder scope.
Evaluate only the last of three full epochs; do not select a best model on test.
"""
_base_ = ['c1_3e.py']
geometry_loss_weight = 0.10
