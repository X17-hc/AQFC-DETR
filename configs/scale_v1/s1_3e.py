"""S1: C0 recipe, changing only the post-composition training transform.

Engineering trainval/test diagnostic; not an unbiased research comparison.
Mosaic/Copy-Paste still follow the inherited fixed phase-23 schedule.
"""
_base_ = ['../geometry_v1/c0_3e.py']
train_transform_mode = 'native800'
