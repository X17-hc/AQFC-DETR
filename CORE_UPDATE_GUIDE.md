# 核心优化与实验工具更新使用说明

更新日期：2026-09-07。默认训练配置保持不变；新算法仅在独立实验配置启用。

## 1. 在 PyCharm 中手动启动

解释器继续使用 `D:/venv/AQFC-DETR/Scripts/python.exe`，工作目录为项目根目录。
已保留原三个配置，并新增：

| 配置名称 | 内容 |
|---|---|
| AQFC-DETR更新版短训练测试 | 轻量AQBA＋空间候选，20训练step＋2评估batch |
| AQFC-DETR基线完整1个epoch及评估 | 标准方法完整工程对照 |
| AQFC-DETR轻量AQBA完整1个epoch及评估 | 单独验证轻量分支 |
| AQFC-DETR空间候选完整1个epoch及评估 | 单独验证空间候选 |
| AQFC-DETR更新版完整1个epoch及评估 | 组合变体完整工程检查 |

配置保存于 `.run`，PyCharm运行配置下拉菜单可选择。若IDE未刷新，重新打开项目后查看。
五个新增配置全部是手动入口，没有启动前训练任务、定时器或自动串行实验。

共同设置：batch=1、workers=0、AMP开启、256/288/320训练尺度、max_size=384、关闭EMA/Mosaic/Copy-Paste、seed42。
工程检查加载旧best305、使用trainval→test；完整一轮为epochs=1、val_epoch=[0]、训练/评估步数限制均为0。
每次运行在各自输出根目录下创建时间戳子目录，原文件不会覆盖。续训不要使用`--unique-output-dir`。

建议先手动运行“更新版短训练测试”，检查成功后再手动选择完整一轮。
**本次交付没有启动真实数据训练或完整验证；一轮训练的AP不是成熟模型精度结论。**

## 2. 配置矩阵

`configs/experiments/local8gb`为工程版本，`configs/experiments/research`为train→val研究版本；两者目前都采用8GB尺度。

| 名称 | 密度目标 | AQBA编码器 | 候选 |
|---|---|---|---|
| baseline | reference | standard | fused |
| target_fast | vectorized | standard | fused |
| aqba_light | vectorized | light_dw | fused |
| spatial | vectorized | standard | spatial |
| light_spatial | vectorized | light_dw | spatial |
| modules_off_900 | 不执行 | 不构建 | semantic固定900，DGFC关闭 |

轻量分支保持256通道、原网格和既有计数/边界/密度头，6层DW+PW的dilation为1/2/3/1/2/1，GN32，不新增SE、池化或NMS。
空间模式保留ceil(0.75K)语义候选，按8×8有效区域密度分配其他名额；最大余数法带容量限制，稳定排序，同分以token索引打破。
密度失效时显式退化语义选择并统计；旧semantic/fused/mixed模式不变。

研究启动示例（仅供手动执行）：

```powershell
D:/venv/AQFC-DETR/Scripts/python.exe main.py --config configs/experiments/research/light_spatial.py --no-pretrained --output-dir outputs/research_light_spatial --unique-output-dir --num_workers 0
```

研究配置必须使用`--no-pretrained`，保留ImageNet Backbone初始化。代码不接受研究配置悄悄使用默认best305。
相同协议分别运行baseline、target_fast、aqba_light、spatial、light_spatial；基线与最终候选再用`--seed 42/43/44`分别启动。
如需正式大尺度训练，在有足够显存的机器上另建同尺度成对配置；不可把本地低分辨率结果与外部800像素模型直接比较。

## 3. 记录与权重

每次启动生成`experiment_manifest.json`：源码状态、解析配置、数据与权重哈希、类别、初始化来源、环境、用途、模型签名；训练后更新完整epoch标志。
未知历史权重的训练来源仍标为unknown，不把哈希当作数据无泄漏证明。

- `--pretrained`：允许旧权重迁移与部分加载；轻量卷积独立键名并重新初始化。
- `--resume`：训练恢复时核对签名。跨变体/策略切换必须warm-start。
- 无签名的历史模型只允许原标准路径按旧规则恢复。
- 覆盖率报告同时保留state_dict（含buffer）与parameter-only口径。
- 不自动提交、推送或升级默认配置。

## 4. 预测导出和诊断

正常评估追加`--export-predictions`，会生成独立时间戳预测目录：

- `predictions.json`：原image_id/category_id，像素xywh和score，无新增展示阈值。
- `images.jsonl`：包含空预测图像、有效查询数、图像尺寸。
- `metadata.json`：评估图像清单、类别映射和配置。

普通评估不额外逐图同步，因此导出的latency_ms为null，明确标注未测量；需要时间统计使用benchmark工具。
追加`--export-diagnostics`时，每个rank最多保存32张图像的密度图、有效网格和Encoder proposal数组；路径在images.jsonl中。
未知类别（例如未映射的第9通道）会报错，不静默重新编号。原始AI-TOD标注保持0–7，不调用Supervision导出覆盖数据集。

离线诊断示例：

```powershell
python tools/analyze_detection_errors.py --annotations <标注JSON> --predictions <predictions.json> --metadata <metadata.json> --output-dir <新分析目录> --image-root <图像目录> --diagnostics-dir <diagnostics目录>
```

默认confidence=.25、IoU=.5，生成类别TP/FP/FN/duplicate、尺寸和密度分桶、混淆矩阵及图像叠加。
白点表示候选中心，密度覆盖图与proposal_recall.json仅针对已导出的有限诊断图像。
crowd/ignore GT在该诊断统计中排除；这不是官方COCO匹配替代，也不是LRP阈值计算。`no_gt`与`no_tp_at_diagnostic_threshold`不能解释为模型训练崩溃。

## 5. 切片推理（手动独立工具）

```powershell
python tools/evaluate_sliced.py --config <模型配置> --checkpoint <同结构AQFC权重> --annotations <标注JSON> --image-root <图像目录> --output-dir <新输出目录> --image-ids <JSON图像ID列表>
```

默认320切片、64重叠、单线程单切片、class-aware NMS/IoU=.5，不追加全图分支。
可用`--slice-wh 480 --overlap-wh 96`，或`--include-full-image`，或`--merge nmm --overlap-metric ios`做独立对照。
不提供image-ids会遍历全部标注图像，只应由用户明确手动执行。
所有分支的查询数、前向时间、端到端时间和合并时间分别记录，最终使用原AI-TOD评估。
不保证切片加速，通常需要多次模型调用；不能把切片结果放进单次推理效率表。

## 6. 性能分析

手动短profile：

```powershell
python tools/profile_training_pipeline.py --config configs/experiments/local8gb/light_spatial.py --data-root <AITOD根目录> --pretrained weights/legacy/dqdetr_best305.pth --output-dir outputs/profile_update --steps 20
```

该工具仅在用户执行时启动真实短训练，不运行验证。steps限制1–100。
生成Chrome trace及summary：DataWait、H2D、Backbone、Encoder、AQBA、DensityPyramid、DGFC、CandidateSelection、Decoder、Matcher/Criterion、密度损失、Backward、Optimizer。
嵌套范围时间不能相加，profile耗时不能当作无插桩训练速度；普通训练没有profile hooks和额外CUDA同步。

动态查询benchmark支持`--amp/--no-amp`及`--images <清单JSON>`。
真实清单格式为`[{"path":"D:/.../image.jpg","gt_count":123}]`；gt_count仅用于事后密度分桶，不参与模型。
真实图像模式目前限定batch=1，测模型和端到端耗时（端到端不含文件读取/解码，包含预处理、传输和后处理）。
合成模式继续支持显式batch2/4。默认50次预热、200次计时；本次没有运行长benchmark。
输出包含input_size、AMP、查询数、allocated/reserved显存与密度分组结果。

## 7. 可选依赖与快速评估状态

Supervision 0.30.1已安装，新增依赖锁在`requirements-analysis-lock.txt`。训练入口不导入Supervision。
torch2.7.1+cu118、torchvision0.22.1+cu118、NumPy1.26.4、OpenCV4.11.0.86保持原版本，pip check通过。

`faster-coco-eval-aitod==1.0.2`未安装：官方只有cp39/manylinux wheel与源码包，当前Windows/Python3.11没有兼容wheel。
未尝试冒险编译或升级训练环境。`requirements-fast-eval.txt`只是候选版本记录，不是本机已验收的依赖。

默认`eval_backend='legacy'`不变。显式`faster_aitod`若依赖缺失会清楚报错。
适配器当前为双后端一致性验证，仍完整运行legacy以保留LRP和原表格，比较precision/recall张量及独立AP25，阈值1e-6。
**它不是已验收的加速实现；双算通常更慢。** 完整快速替代需要在支持环境通过一致性测试后再实现/启用，不能宣称本次已完成评估提速。

固定预测比较工具：

```powershell
python tools/compare_eval_backends.py --annotations <GT> --predictions <预测JSON> --metadata <metadata.json> --output <新报告JSON>
```

## 8. 当前验收边界

自动验证只使用单元fixture与合成图像，不读取真实数据训练、不跑完整验证、不跑多日实验。
整模型合成测试使用128×128：64×64在batch1下使原DGFC末层变成1×1，触发原有BatchNorm限制，未为此改变默认模型。
真实一轮运行、AP增益、训练吞吐提升、实际推理提速均待用户手动实验。
新方法即使达标，也不会自动替换默认方法。
