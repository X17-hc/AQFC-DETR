"""Single-knob joint DFL 0.25 -> 0.35. o2m stays 0.25.
Contrast H1 32.2/15.6. Gate +0.3 AP and +0.3 APvt, AP75>=-0.2pp.
32/1000 precheck first. geometry_loss_weight stays 0. Do not also change o2m.
"""
_base_ = ['h1_finetune_3e.py']
joint_dfl_coef = 0.35
joint_o2m_coef = 0.25
