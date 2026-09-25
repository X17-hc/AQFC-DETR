"""S2 engineering comparison: same C0 start/recipe, mild scales without crop.

Only the post-composition transform differs from S1. Evaluation stays unchanged.
The mode fixes [640, 704, 768, 800] / max1333; data_aug_scales does not override it.
"""
_base_ = ['./s1_3e.py']
train_transform_mode = 'native_multiscale'
