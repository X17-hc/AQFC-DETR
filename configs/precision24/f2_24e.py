"""F2: preserve six full encoder layers; isolate local box refinement.

24-epoch schedule even for the first-epoch launch. Never load a P2 adapter.
"""
_base_ = ['./f0_24e.py']
detail_context_encoder = False
local_refiner_enabled = True
precision24_stage_override = None
p2_transfer_version = None
p2_transfer_mode = 'reference'  # Neutral default, no teacher/reference module created.
val_epoch = [0, 23]
save_checkpoint_interval = 1

