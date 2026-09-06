# 迁移说明

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
