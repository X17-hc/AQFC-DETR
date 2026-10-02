"""3-epoch box-head finetune that reweights near-miss boxes.

Matched boxes with IoU in [0.5, 0.75) and original-pixel area at least 64
get 3x GIoU and width/height loss. Verytiny boxes and already-tight boxes
keep weight 1. Encoder, density gate, and query allocator stay frozen.
"""
_base_ = ['h1_ap_box_control_3e.py']
box_tighten = True
