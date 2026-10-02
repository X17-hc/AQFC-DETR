"""H1 24-epoch recipe with the DGFC density gate and AQBA quantile boundaries.

Only switches that gained on the paired 1-epoch probe should stay enabled
before a full run. This file turns both on for the case where both gain.
"""
_base_ = ['h1_24e.py']
calibrator_density_spatial = True
allocator_quantile_boundaries = True
