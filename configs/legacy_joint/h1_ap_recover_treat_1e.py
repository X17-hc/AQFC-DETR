"""1-epoch decoder-head run that doubles the near-miss box loss.

Verytiny matches stay at weight 1. Area >= 64 and IoU in [0.5, 0.75)
scale GIoU and width/height by 2. Center L1 and NWD stay unscaled.
"""
_base_ = ['h1_ap_recover_control_1e.py']
box_tighten = True
box_tighten_weight = 2.0
