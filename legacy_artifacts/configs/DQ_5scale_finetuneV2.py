# config/DQ_5scale_finetuneV3.py
"""
Fine-tune 阶段 3 配置（修复版）
────────────────────────────────────────────────────────────────────────
起点    : 0.315 AP 的最佳 checkpoint
上一版  : 0.302 AP（Mosaic+CopyPaste叠加 → FP暴增）
本版修复:
  1. Mosaic 与 Copy-Paste 互斥（总增强概率 = mosaic_p + copy_paste_p = 0.6）
  2. Mosaic 使用 warmup + cosine decay 调度（初期不猛触发）
  3. epochs 增至 18（给模型更多时间适应 Mosaic 分布再收敛）
  4. lr_drop 节点随之调整
  5. mosaic_p 降至 0.3（上一版 0.5 过强）

使用方法：
  python main_aitod.py \
      -c config/DQ_5scale_finetuneV3.py \
      --pretrain_model_path logs/AQFCDETR_ver6_CopyPaste/0330-NewRecord/checkpoint0023.pth \
      --output_dir logs/AQFCDETR_ver7/0412-fixedMosaic \
      --dataset_file aitodv2 \
      --coco_path /workspace/DQDetr/data/path/AITODv2 \
      --amp

注意：用 --pretrain_model_path 不用 --resume（让 lr_scheduler 重置）。
"""

_base_ = ['coco_transformer.py']

# ── 数据增强尺度 ──────────────────────────────────────────────────────
data_aug_scales = [480, 512, 544, 576, 608, 640, 672,
                   704, 736, 768, 800, 864, 900]
data_aug_max_size       = 1333
data_aug_scales2_resize = [400, 500, 600, 700]
data_aug_scales2_crop   = [384, 600]

# ── Mosaic 参数（互斥版，比上一版保守）──────────────────────────────
mosaic_p            = 0.3    # 峰值触发概率（上一版 0.5 → 改为 0.3）
mosaic_center_ratio = (0.35, 0.65)
mosaic_fill_value   = 114

# Mosaic 概率调度（在 engine.py 的 MosaicPScheduler 中使用）
mosaic_warmup_end   = 2      # epoch 0~1 线性 warmup
mosaic_decay_start  = 14     # epoch 14 开始余弦退出

# Copy-Paste 触发概率（互斥分配）
# 总增强概率 = mosaic_p + copy_paste_p = 0.3 + 0.3 = 0.6
# 纯净样本概率 = 0.4（上一版纯净样本几乎为 0）
copy_paste_p = 0.3           # 上一版隐含 0.5 → 改为 0.3

# ── 基本训练参数 ──────────────────────────────────────────────────────
num_classes = 9
batch_size  = 2
epochs      = 18             # 上一版 12 → 改为 18（充分收敛）

# ── 学习率 ────────────────────────────────────────────────────────────
lr          = 1e-5
lr_backbone = 1e-6

param_dict_type       = 'default'
lr_backbone_names     = ['backbone.0']
lr_linear_proj_names  = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult   = 0.1
ddetr_lr_param        = False

weight_decay  = 0.0001
clip_max_norm = 0.1
onecyclelr    = False
multi_step_lr = True

# lr 衰减节点（18 epoch）：
#   epoch 0-11 : lr = 1e-5
#   epoch 12-16: lr = 1e-6
#   epoch 17   : lr = 1e-7
lr_drop_list = [12, 17]
lr_drop      = 12

val_epoch              = [17]   # 多次评估，选最优 EMA checkpoint
save_checkpoint_interval = 1

# ── DN 配置 ───────────────────────────────────────────────────────────
use_dn               = True
dn_number            = 100
dn_box_noise_scale   = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt       = False
dn_labelbook_size    = 91
match_unstable_error = True

# ── EMA ──────────────────────────────────────────────────────────────
use_ema   = True
ema_decay = 0.9999
ema_epoch = 0

# ── 其余参数（与 stage 1 完全相同）──────────────────────────────────
ccm_params          = [8, 16, 32]
ccm_cls_num         = 4
dynamic_query_list  = [300, 500, 900, 1500]
find_unused_parameters = False
modelname           = 'dqdetr'
frozen_weights      = None
backbone            = 'resnet50'
use_checkpoint      = False
dilation            = False
position_embedding  = 'sine'
pe_temperatureH     = 20
pe_temperatureW     = 20
return_interm_indices = [0, 1, 2, 3]
backbone_freeze_keywords = None
enc_layers          = 6
dec_layers          = 6
unic_layers         = 0
pre_norm            = False
dim_feedforward     = 2048
hidden_dim          = 256
dropout             = 0.0
nheads              = 8
num_queries         = 900
query_dim           = 4
num_patterns        = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW        = -1
random_refpoints_xy = False
fix_refpoints_hw    = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder      = False
dabdetr_deformable_decoder      = False
use_deformable_box_attn         = False
box_attn_type       = 'roi_align'
dec_layer_number    = None
num_feature_levels  = 5
enc_n_points        = 4
dec_n_points        = 4
decoder_layer_noise = False
dln_xy_noise        = 0.2
dln_hw_noise        = 0.2
add_channel_attention    = False
add_pos_value            = False
two_stage_type           = 'standard'
two_stage_pat_embed      = 0
two_stage_add_query_num  = 0
two_stage_bbox_embed_share  = False
two_stage_class_embed_share = False
two_stage_learn_wh          = False
two_stage_default_hw        = 0.05
two_stage_keep_all_tokens   = False
num_select               = 300
transformer_activation   = 'relu'
batch_norm_type          = 'FrozenBatchNorm2d'
masks                    = False
aux_loss                 = True

set_cost_class   = 2.0
set_cost_bbox    = 5.0
set_cost_giou    = 2.0
cls_loss_coef    = 1.0
mask_loss_coef   = 1.0
dice_loss_coef   = 1.0
bbox_loss_coef   = 5.0
giou_loss_coef   = 2.0
enc_loss_coef    = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha      = 0.25

decoder_sa_type      = 'sa'
matcher_type         = 'HungarianMatcher'
decoder_module_seq   = ['sa', 'ca', 'ffn']
nms_iou_threshold    = -1
dec_pred_bbox_embed_share  = True
dec_pred_class_embed_share = True
use_detached_boxes_dec_out = False

coverage_loss_coef = 0.1
interval_loss_coef = 0.05
spacing_loss_coef  = 0.2
ccm_loss_coef      = 1.0
nwd_loss_coef      = 1.0