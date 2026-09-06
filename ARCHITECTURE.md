# AQFC-DETR 架构说明

## 1. 主数据流

```text
image → backbone multi-level features → deformable encoder memory
      → AQBA: count / ordered boundaries / density prior / query budget
      → DGFC: channel calibration + density-guided spatial residual
      → encoder semantic logits + proposal boxes
      → semantic-density proposal ranking
      → grouped dynamic decoder
      → class logits and refined boxes
```

AQBA 使用最高分辨率 encoder level，但输出的密度图会插值至全部 feature levels。密度图既进入 DGFC 空间残差，也进入 proposal 排序，并由 GT 中心 Gaussian 热图直接监督，因此不是与检测主链路分离的旁路特征。

## 2. AQBA

`AdaptiveQueryBudgetAllocator` 从 `[B,S,256]` memory 还原 level 0 特征，预测真实目标数、三条有序路由边界、四档分类 logits 和空间密度图。边界在对数空间构造：

```text
log b2 = log b1 + softplus(delta12) + 0.3
log b3 = log b2 + softplus(delta23) + 0.3
```

初始/引导边界为 60、150、350 个路由计数。训练路由采用确定性教师退火：`N_route = r*(1.5*N_gt+50)+(1-r)*N_pred`；epoch 0–5 的 r 为 1/1/1/.75/.5/.25，之后为 0。推理始终只用预测计数，不设 900 下限。非有限预测直接选择 900 档。

密度目标在 level 0 网格生成，中心 Gaussian 半径为 `clamp(round(.5*max(wW,hH)),1,4)`，重叠目标逐像素取最大。损失包括 coverage、spacing、count、interval、boundary guide 与 density focal，各权重仅由配置控制。

## 3. DGFC

`DensityGuidedFeatureCalibrator` 的主门控为：

```text
z = MLP(GAP(F)) + MLP(GMP(F))
F_channel = F * (1 + tanh(clamp(s,0,.2) * z))
F_out = F_channel * (1 + alpha_i * SpatialAttention(D_i))
```

零响应严格对应单位乘数，既可增强也可抑制。level 0 的空间系数为 0，避免最高分辨率特征受密度旁路重复扰动；通道校准仍覆盖所有层。

## 4. Proposal 选择

联合分数为 `semantic_logit + 0.25*density_logit`。Top-K 索引同时用于提取 encoder memory 和对应的 4D proposal，密度只负责排序，不制造固定宽高参考框。mixed 消融先取 75% 语义候选，再取去重后的 25% 密度候选，不足部分以联合分数补齐。

## 5. 动态 Decoder 与 mask

训练时 DN 查询不能按样本拆 batch，因此匹配查询 padding 到 batch 最大 K，并把 `~query_valid_mask` 作为 target key padding mask。Matcher、Criterion 和 PostProcess 均再次读取该 mask。

推理无 DN：相同预算样本组成一组，每组只选择 K 个 proposal 并调用一次 Decoder，随后散射、padding 回 batch 最大 K。记录 `sum(K_i)`、`B*max(K_i)` 和实际 token 降幅。关闭 `grouped_decoder_inference` 时回退为 batch-max 执行。

## 6. 配置回退

- `proposal_density_weight=0`：纯语义 Top-K；
- `calibrator_use_spatial=False`：仅通道校准；
- `grouped_decoder_inference=False`：batch-max Decoder；
- `force_query_budget=900`：固定 900 查询基线。
