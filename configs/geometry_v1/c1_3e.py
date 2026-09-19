"""C1 differs from C0 ONLY by last-decoder matched geometry supervision.

Weight ramps from zero by successful optimizer updates during one loader-length
of training, then remains 0.05. AMP skipped updates do not advance the ramp.
"""
_base_ = ['c0_3e.py']
geometry_loss_weight = 0.05
