"""24 new epochs initialized by the original paper detector, NOT epoch10 resume.

Use the dedicated launcher. Partial state migration is intentional because the
light density encoder has different keys; the source file remains SHA-256 pinned.
"""
_base_ = ['p2_24e.py']

# The legacy file contains only model state, no AQFC training phase/signature.
# Start the new-module teacher/loss/augmentation curriculum from zero.
strict_warmstart = False
expected_pretrained_epoch = None
expected_pretrained_sha256 = 'f854eec1f0d0ebf6e131d01e0b2826ce5690b593a5ed7f3fb8203453a62936e4'
training_phase_epoch_offset = 0
training_phase_total_epochs = 24

# Use the existing server 24-epoch adaptation recipe, not the much lower LR
# used for a three-epoch fine-tune of already-trained AQFC light modules.
lr = 1e-4
lr_backbone = 1e-5
multi_step_lr = True
onecyclelr = False
lr_drop_list = [13, 23]
# Inherit: quality_blend=0.25, one-epoch quality warmup, EMA off, 24 epochs,
# checkpoint every epoch, final-only evaluation, and engineering optimizations.
