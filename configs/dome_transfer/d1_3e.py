"""Combination experiment; its difference from D0 is not a single-module ablation."""
_base_ = ['d0_3e.py']
dome_transfer_recipe = 'd1'
density_underestimate_weight = 0.10
proposal_selection_mode = 'protected_density'

