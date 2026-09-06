torchrun --nproc_per_node=1 main_aitod.py \
  --output_dir logs/AQFCDETR_ver11/0510-ablation_cfge-sigmoid-full -c config/DQ_5scale_fullV2.py \
  --coco_path /workspace/DQDetr/data/path/AITODv2 \
  --pretrain_model_path /workspace/DQDetr/models/dqdetr_best305.pth \
  --options dn_scalar=100 embed_init_tgt=False \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0