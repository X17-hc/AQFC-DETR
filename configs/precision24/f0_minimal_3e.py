"""Minimal F0 control: three complete epochs of the unchanged 24-epoch recipe.

Launch with --stop-after-epochs 3. Evaluation is a separate subset-only step
after the real checkpoint path is known; do not run full test automatically.
"""
_base_ = ['./f2_24e.py']

# Only algorithmic difference from F2: no local refiner, hence original final
# regression and original-box quality target. Preserve six full encoder layers.
local_refiner_enabled = False

# Keep epochs=24 and LR milestones 16/22 inherited. Runtime stopping at epoch2
# must not redefine the LR horizon or restart the original phase23 curriculum.
val_epoch = []
save_checkpoint_interval = 1
