"""F0: same S2 ordinary checkpoint, fresh optimizer, 24 full engineering epochs."""
_base_ = ['../scale_v1/s2_3e.py']
precision24_enabled = True
detail_context_encoder = False
local_refiner_enabled = False
precision24_stage_override = None
precision24_new_lr = 1e-4
precision24_warmup_updates = 500
epochs = 24
val_epoch = [23]
lr = 3e-5
lr_backbone = 3e-6
weight_decay = 1e-4
clip_max_norm = 0.1
multi_step_lr = True
lr_drop_list = [16, 22]
expected_pretrained_epoch = 2
expected_pretrained_sha256 = '40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928'
