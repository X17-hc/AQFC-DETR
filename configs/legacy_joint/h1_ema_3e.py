"""EMA 3e from H1 epoch23. Evaluate the EMA weights only.
Contrast H1 32.2/15.6. Gate +0.3 AP and +0.3 APvt, AP75>=-0.2pp.
32/1000 precheck first. geometry_loss_weight stays 0. No F2 overlay.
"""
_base_ = ['h1_finetune_3e.py']
joint_allow_ema = True
use_ema = True
ema_decay = 0.9997
