"""Only AQBA/target/candidate variant differs from the server baseline."""
_base_ = ['aitodv2_standard_24e.py']
allocator_encoder_type = 'light_dw'
density_target_backend = 'vectorized'
proposal_selection_mode = 'spatial'
spatial_semantic_ratio = 0.75
spatial_grid_size = [8, 8]
density_target_chunk_size = 512
