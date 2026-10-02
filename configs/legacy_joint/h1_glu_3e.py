"""DGFC GLU 3e on H1 epoch23. Swish stays closed.
Contrast H1 32.2/15.6. Gate +0.3 AP and +0.3 APvt, AP75>=-0.2pp.
32/1000 precheck first. geometry_loss_weight stays 0. Do not reopen Swish.
"""
_base_ = ['h1_finetune_3e.py']
calibrator_gate_type = 'glu'
