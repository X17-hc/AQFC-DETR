"""P1: same repaired model and recipe, numerical-equivalent engineering switches."""
_base_ = ['repaired.py']
aligned_box_loss = True
batched_metric_transfer = True
non_blocking_transfer = True
