"""F1: joint encoder/detail refinement; not an isolated single-module ablation."""
_base_ = ['./f0_24e.py']
detail_context_encoder = True
local_refiner_enabled = True
