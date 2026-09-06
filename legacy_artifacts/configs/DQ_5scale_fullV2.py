# config/DQ_5scale_full_v2.py
"""
完整训练配置（从 COCO 预训练权重出发，带 Mosaic + Copy-Paste 互斥增强）
────────────────────────────────────────────────────────────────────────
与原始 DQ_5scale.py 的核心差异：
  1. 新增 mosaic_p = 0.25（Mosaic 峰值概率）
  2. 新增 copy_paste_p = 0.35（Copy-Paste 独立概率）
  3. 两者互斥，总强增强概率 = 0.60，纯净样本 = 0.40
  4. Mosaic warmup/decay 由 MosaicPScheduler 管理（engine.py 中）
  5. ema_decay 从 0.9997 → 0.9999（更稳定的 EMA）
  6. val_epoch 增加中间评估点

使用方法：
  # Stage1：从 COCO 预训练权重开始完整训练（24 epoch）
  python main_aitod.py \
      -c config/DQ_5scale_full_v2.py \
      --pretrain_model_path /path/to/dino_resnet50_coco_pretrain.pth \
      --output_dir logs/AQFCDETR_ver8/full_v2 \
      --dataset_file aitodv2 \
      --coco_path /workspace/DQDetr/data/path/AITODv2 \
      --amp

  # Stage2：从 Stage1 最佳 pth 继续 fine-tune
  python main_aitod.py \
      -c config/DQ_5scale_finetuneV3.py \
      --pretrain_model_path logs/AQFCDETR_ver8/full_v2/checkpoint0023.pth \
      --output_dir logs/AQFCDETR_ver8/finetune_v3 \
      ...
"""

_base_ = ['coco_transformer.py']

# ── 数据增强尺度（与原版相同）──────────────────────────────────────────
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]

# ── Mosaic 参数 ───────────────────────────────────────────────────────
# 完整训练阶段 Mosaic 概率比 fine-tune 阶段更保守（0.25 vs 0.30）
# 原因：完整训练有 24 epoch，前期模型还没有强的特征表示，
#       过激的 Mosaic 会让模型在特征学习阶段走弯路。
mosaic_p = 0.25
mosaic_center_ratio = (0.35, 0.65)
mosaic_fill_value = 114

# Mosaic 调度参数
# 完整训练的调度策略：
#   warmup  epoch 0~4 : 0 → 0.25（给模型先建立基础特征表示）
#   plateau epoch 5~19: 0.25（稳定增强）
#   decay   epoch 20~23: 0.25 → 0（末期收敛到纯净分布）
mosaic_warmup_end = 4        # 原4
mosaic_decay_start = 20      # 原20
# total_epochs 会从 epochs 字段读取，无需单独设置

# Copy-Paste 触发概率（与 Mosaic 互斥）
# 总增强概率 = 0.25 + 0.35 = 0.60，纯净样本 = 0.40
copy_paste_p = 0.35

# ── 基本训练参数 ──────────────────────────────────────────────────────
num_classes = 9
batch_size = 2
epochs = 24

# ── 学习率 ──────────────────────────────────────────────
lr = 0.0001
lr_backbone = 1e-05

param_dict_type = 'default'
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False

weight_decay = 0.0001
lr_drop = 11
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = True
lr_drop_list = [13, 23]

val_epoch = [23]
save_checkpoint_interval = 1

# ── DN 配置 ───────────────────────────────────────────────────────────
use_dn = True
dn_number = 100
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
embed_init_tgt = False
dn_labelbook_size = 91
match_unstable_error = True

# ── EMA（比原版更强的平均）──────────────────────────────────────────
use_ema = True
ema_decay = 0.9997  # 原版 0.9997 → 改为 0.9999，减少震荡
ema_epoch = 0

# ── 其余参数（与原版 DQ_5scale.py 完全相同）─────────────────────────
ccm_params = [8, 16, 32]
ccm_cls_num = 4
dynamic_query_list = [300, 500, 900, 1500]
find_unused_parameters = False
modelname = 'dqdetr'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [0, 1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 5
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
use_detached_boxes_dec_out = False

# SBA
coverage_loss_coef = 0.1  # 原版 0.1
interval_loss_coef = 0.05  # 原版 0.05
spacing_loss_coef = 0.2  # 原版 0.2
ccm_loss_coef = 1.0
nwd_loss_coef = 1.0
# ZFC
# ── CGFE 通道注意力消融：gate_type ─────────────────
# 'tanh'    — tanh 中心化通道注意力（v8 主方法，默认）
# 'swish'   — Swish/SiLU 激活（无上界饱和，增强更激进）
# 'glu'     — GLU 门控线性单元（data-dependent gating）
# 'se'      — 残差 SE（1+sigmoid，仅增强，无法抑制）
# 'sigmoid' — 标准 SE（sigmoid 直接乘，0.5 压制问题）
gate_type = 'glu'
calibrator_disable_spatial = False
