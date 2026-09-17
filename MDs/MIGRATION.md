# 迁移说明

## 核心优化变体（2026-09-07）

标准分支保留旧state_dict结构和默认选择。`light_dw`使用独立`light_density_encoder`键，原密集卷积不转换成DW/PW。
新checkpoint记录`variant_signature`，训练resume核对结构、候选、密度目标、损失与数据协议；不一致时使用warm-start。
无签名历史checkpoint只允许原标准路径按既有规则恢复，新变体必须`--pretrained`。
迁移报告新增parameter-only覆盖率，与既有含buffer覆盖率分开。
`--unique-output-dir`仅用于新实验，与`--resume`冲突。完整操作见`CORE_UPDATE_GUIDE.md`。

## 名称映射

| 历史名称 | AQFC-DETR 名称 |
|---|---|
| DQ-DETR 衍生模型 / `DQDETR` | AQFC-DETR / `AQFCDETR` |
| CCM / `AdaptiveBoundaryCCM` | AQBA / `AdaptiveQueryBudgetAllocator` |
| `TrueAdaptiveBoundaryLoss` | `QueryBudgetLoss` |
| CGFE / `CGFE` | DGFC / `DensityGuidedFeatureCalibrator` |
| `MultiScaleFeature` | `DensityPyramidAdapter` |

活动包入口为 `from models.aqfcdetr import build_aqfcdetr`，配置注册名固定为 `modelname = "aqfcdetr"`。

## Checkpoint 规则

历史 checkpoint 必须通过 `--pretrained` 加载。加载器会提取 `model`/`ema_model`、移除 DDP 前缀、映射历史模块名、跳过形状不一致参数，并在输出目录生成 `checkpoint_migration_report.json`。报告按参数量计算覆盖率。

`--resume` 只接受 AQFC-DETR 新 checkpoint，并严格恢复模型；若包含历史模块键会提示改用 `--pretrained`。

```powershell
python tools/convert_legacy_checkpoint.py `
  --input weights/legacy/dqdetr_best305.pth `
  --output weights/converted/aqfc_warmstart.pth
```

转换结果只含初始化权重，不伪造 optimizer、scheduler 或 epoch。历史配置字段不兼容，`tools/validate_config.py` 会报告对应的新字段。

历史日志和路径未重写，统一保存在 `legacy_artifacts/`，以维护实验记录完整性。
