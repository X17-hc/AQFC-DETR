"""24-epoch decoder-head recovery from the same scheme-2 epoch23 weights.

Start again from checkpoint0023. The learning rate stays at 1e-5.
"""
_base_ = ['h1_ap_recover_treat_1e.py']
epochs = 24
