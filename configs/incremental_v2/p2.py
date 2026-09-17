"""P2: P1 plus the preregistered decoder-only quality blend; no inference head."""
_base_ = ['p1.py']
classification_loss_type = 'quality_blend'
