"""One-epoch probe config: DGFC density spatial gate only."""
_base_ = ['h1_24e.py']
calibrator_density_spatial = True
allocator_quantile_boundaries = False
