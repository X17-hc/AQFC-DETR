"""Frozen six-layer reference. Use the dedicated trainer for adaptation."""
_base_ = ['../precision24/f0_24e.py']
local_refiner_enabled = True
detail_context_encoder = False
p2_transfer_version = 'layerwise_v2'
p2_transfer_mode = 'reference'
