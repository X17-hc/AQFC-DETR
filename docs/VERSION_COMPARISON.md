# 三方版本核对

采集截止（UTC）：2026-10-23T15:40:03.392509+00:00。对比按SHA-256，并单独将CRLF归一化为LF；没有忽略其他空白或代码差异。

## 当前服务器与本地

服务器294项，本地309项；内容相同284项，实质不同9项。

| 类型 | 文件 |
|---|---|
| 内容不同 | `models/aqfcdetr/joint_modules.py` |
| 内容不同 | `models/aqfcdetr/feature_calibrator.py` |
| 内容不同 | `models/aqfcdetr/model.py` |
| 内容不同 | `util/incremental.py` |
| 内容不同 | `util/legacy_joint.py` |
| 内容不同 | `util/checkpoint.py` |
| 内容不同 | `tests/test_joint_integration.py` |
| 内容不同 | `engine.py` |
| 内容不同 | `main.py` |
| 仅服务器 | `tools/diagnose_p2_memory_swap.py` |
| 仅本地 | `configs/improvement_v1/BQ_1e.py` |
| 仅本地 | `configs/improvement_v1/BQ_3e.py` |
| 仅本地 | `configs/improvement_v1/B_1e.py` |
| 仅本地 | `configs/improvement_v1/B_3e.py` |
| 仅本地 | `configs/improvement_v1/Q_1e.py` |
| 仅本地 | `configs/improvement_v1/Q_3e.py` |
| 仅本地 | `configs/improvement_v1/R_1e.py` |
| 仅本地 | `configs/improvement_v1/R_3e.py` |
| 仅本地 | `models/aqfcdetr/improvement_modules.py` |
| 仅本地 | `tests/test_improvement_v1.py` |
| 仅本地 | `tools/benchmark_improvement.py` |
| 仅本地 | `tools/check_improvement_config.py` |
| 仅本地 | `tools/precheck_improvement.py` |
| 仅本地 | `tools/run_improvement_ladder.py` |
| 仅本地 | `util/improvement.py` |
| 仅本地 | `util/improvement_analysis.py` |

## 与冻结实验的关系

原180项清单逐字节全部匹配。服务器当前主目录在同一清单范围内177项相同，缺少 `configs/mature_pretrain24.py`、`configs/pretrain_smoke.py`，`util/legacy_joint.py` 不同。

本地在同一范围内90项逐字节相同、75项仅换行不同、8项实质不同、7项缺失（两个专用配置和五个构建产物/副本）。8个差异文件为：

- `models/aqfcdetr/joint_modules.py`
- `models/aqfcdetr/feature_calibrator.py`
- `models/aqfcdetr/model.py`
- `util/incremental.py`
- `util/legacy_joint.py`
- `util/checkpoint.py`
- `engine.py`
- `main.py`

冻结版本的权重加载代码识别 `pretrain_model.pth` 的SHA-256，并允许对应22个额外缺失参数；服务器当前主目录缺少该特例。发布采用冻结版本。

## 发布范围

175项源码原字节保留；5项构建产物/副本不发布。依赖构建资料来自父提交，并与当前服务器评估器源码交叉核对。旧工具、旧实验报告和其他历史输出从此基线分支的当前树移除，在父提交和main中保留。

本次未改写服务器源码和原本地工作区，未执行训练、推理或安装。当前环境版本与历史训练环境分开表述；最终检查点仅核对存在性和文件属性，不宣称权重逐字节校验已完成。
