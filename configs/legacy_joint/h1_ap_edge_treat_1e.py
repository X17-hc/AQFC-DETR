"""1-epoch distribution-refiner treatment from the scheme-2 epoch23 weights.

Log-width smooth L1 applies to refined boxes whose IoU is below 0.60.
"""
_base_ = ['h1_ap_edge_control_1e.py']
edge_logwh = True
