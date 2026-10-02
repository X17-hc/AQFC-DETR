"""One-epoch probe config: AQBA quantile boundaries only."""
_base_ = ['h1_24e.py']
calibrator_density_spatial = False
allocator_quantile_boundaries = True
