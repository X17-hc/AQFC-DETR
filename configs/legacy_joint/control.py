"""No full baseline training launch: initialization/synthetic/short timing only."""
_base_ = ['h1_24e.py']
architecture_variant = 'legacy_joint_control_v2'
joint_subset_epochs = []

