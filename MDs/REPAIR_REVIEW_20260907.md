# AQFC-DETR 修复与复审报告

日期：2026-09-07。目标项目：`D:\PythonProject\AQFC-DETR`。

## 1. 范围与安全边界

本次以用户当前未提交工作树为基准修复，不回滚此前修改，不修改原始 DQ-DETR 项目、数据集和历史日志，不进行 Git 提交或推送。

使用既有的专用环境 `D:\venv\AQFC-DETR\Scripts\python.exe`；没有切换到其他项目环境。本次修正并重新编译安装了项目自带的 AI-TOD COCO API，未更换 PyTorch/CUDA 版本。

修改前文件备份保存在 `outputs/repair_20260907/before/`。回归测试和短训练输出独立保存在 `outputs/repair_20260907/`。没有执行完整论文训练。

## 2. 修复项目与技术理由

### 2.1 半精度数值稳定性

1. **密度损失**：float16/bfloat16 会把 `1-1e-6` 舍入成 1，导致 `log(0)` 和 `0*Inf`。在 FP32 中生成密度概率，并将密度 Focal Loss 的概率、目标与对数运算提升到 FP32；饱和输入的损失和梯度增加回归测试。
2. **数量预测**：先在 FP32 对原始 log-count 限幅，再执行 exp；保留限幅前的有限性标记，避免原始 Inf 被截为 1500 后误认为正常预测。
3. **边界**：边界非线性在 FP32 中执行，限制极端正增量；异常样本在 EMA 更新前被排除。无有效边界时使用已验证的 EMA 或默认有序边界，不让 NaN 污染缓冲区。
4. **Matcher**：分类代价改为 FP32 下基于 softplus 的稳定形式，不再对半精度饱和概率直接取对数；框代价输入使用 FP32。越界类别标签明确报错，不静默改成其他类别。
5. **候选排序**：密度 logit 与语义融合在 FP32 中计算，避免密度先验为 1 时产生 Inf 并丢失语义排序信息。
6. **AMP 观测**：记录 scale、成功更新比例和有限梯度范数；首次非有限梯度列出少量参数名。FP32 非有限梯度和非有限总损失明确失败，整段训练零次成功更新也明确失败。

本地 RTX 4060 配置的初始 AMP scale 设置为实测稳定值 32，GradScaler 仍动态调整。降低 scale 不是代替数值修复；两者都已实施。

### 2.2 初始化与查询路由

1. Transformer 通用 Xavier 初始化结束后，重新应用 AQBA 的专用低方差初始化，且该操作发生在预训练加载之前。不得在加载权重后重置整个 AQBA。
2. 数量回归仍监督真实目标数量的对数。教师和预测路由统一采用 `min(1.5*N+50, max_objects)`，共享 `count_to_routing()`，消除训练/推理尺度不一致。
3. `allocator_use_boundary_ema` 独立控制小规模边界 EMA；原 `use_ema` 继续控制整个模型的 EMA 副本。本地关闭模型 EMA 不再顺带关闭边界平滑。
4. 数值异常的样本回退继续保留；编程错误、形状错误、OOM 等不得被宽泛异常捕获后伪装成另一个正常模型。AQBA/DGFC 异常明确抛出并保留原始原因。
5. 预算上限以合法 encoder proposal 数量为准，而不只统计非 padding token。候选选择排除没有有限参考框的 token；cross-attention 的上下文 padding mask 保持原意。
6. 关闭 DN 不再关闭 AQBA 的 GT 教师输入。训练入口始终传入目标，是否生成 DN 查询由 DN 配置决定。

### 2.3 DN 分支与查询掩码

复审补测发现：`dn_number=0` 原本仍会被提升为一个 DN group，产生额外查询，而后处理又不切除它们，造成 304 个输出与 300 长度 mask 不一致。

现修复为：DN 关闭、推理、全空目标批次直接返回无 DN 查询；DN 张量使用标签嵌入所在设备，不硬编码 `.cuda()`。保留开启 DN 时原有的正负去噪组结构。

新增实际 SetCriterion/PostProcess 测试，确认无效查询的分类梯度为零，空图中的有效背景查询仍学习，padding 框不会进入预测结果。合成 GPU 测试同时覆盖开启/关闭 DN 与四档混合预算。

### 2.4 评估指标

1. 标准评估仍使用 IoU `0.50:0.05:0.95`，保持 AP 与 oLRP 定义不变。
2. AP25 使用单独的 COCO evaluator 累积结果，写回兼容的统计槽位；不会把 0.25 混入标准 AP 平均值。
3. 增加命名字段 `bbox_metrics`，没有有效样本的尺度指标序列化为 JSON `null`、控制台显示 `N/A`。旧数组保留 `-1` 哨兵以兼容读取方，不把无效值当真实负分。
4. 修复自带 COCO API 的 `np.float`，使用 `np.float64`。删除 main.py 中修改 NumPy 全局属性的补丁，独立评估不再依赖入口导入顺序。
5. 查询 token 统计使用总和，均值/分位数独立记录，不再将每 batch 平均 token 数误称为整个评估的 token 总量。

### 2.5 断点、数据划分与日志

1. checkpoint 增加 `format`、`run_metadata`、`best_metrics`；元信息区分短测与完整 epoch，记录实际批次数和数据划分。
2. 短测/不完整 epoch checkpoint 禁止作为完整训练的 `--resume`，但可用 `--pretrained` 初始化，也可通过 `--resume ... --eval` 检查模型。
3. 请求恢复 scaler 或最佳指标历史而断点缺少相应状态时明确报错，不静默重置。已有严格模型、optimizer、scheduler、模型 EMA 恢复保留。
4. regular/EMA 最佳成绩分别维护，checkpoint 在评估后保存，避免恢复时丢掉刚更新的最佳成绩。
5. 测试集或截断评估不用于选择 `checkpoint_best.pth`。训练仍保存普通 `checkpoint.pth`。禁止使用 trainval 训练、val 选模这一重叠组合。
6. 新训练遇到已含 checkpoint 的输出目录明确停止，要求新目录或显式恢复，避免覆盖旧实验。
7. 使用有长度的 LimitedLoader，让短测进度和 ETA 按真实上限显示，也不额外获取一个 batch。
8. 控制台显示核心损失、查询和 AMP 指标；全部损失仍保存在 JSON。详细参数量单独写入 `parameter_counts.json`。Git 状态记录不再调用完整 diff 打印换行符噪声。
9. ResNet 使用显式 ImageNet V1 权重枚举；meshgrid 指定 `indexing='ij'`，不无意改变预训练版本或坐标顺序。
10. 权重迁移报告增加按模块覆盖率，并说明总体覆盖率包含 state_dict 缓冲区，不代表功能等价。

## 3. 验证结果

最终交付验证如下。退出码均为 0；测试过程中发现并修复的失败记录也保留在输出目录，不能用失败中间版本代替下列最终结果。

| 验证项目 | 结果 | 证据 |
|---|---|---|
| 全量单元测试 | **72 passed** | `outputs/repair_20260907/final_pytest.xml` |
| 实验配置 | 19 份通过 | 全部非基础 configs 文件加载并校验 |
| Python 语法 | 109 个文件通过 | 排除历史归档、输出与构建目录的 AST 检查 |
| 专用虚拟环境依赖 | `No broken requirements found` | `python -m pip check` |
| 最终真实短训 | **200 批次、200 次更新、0 次 AMP 跳步** | `release_smoke200/metrics.jsonl` |
| 教师退火结束路径 | **20 批次、20 次更新、0 跳步，teacher_ratio=0** | `teacher_off20/metrics.jsonl` |
| 开启 DN 的合成 GPU 测试 | 通过，epoch=6 | `final_synthetic.json` |
| 关闭 DN 的合成 GPU 测试 | 通过，epoch=0 | `no_dn_synthetic.json` |
| 最密集真实图片 | 1729 个原始标注、1727 个有效标注，1500 查询，一次更新通过 | `dense_training.json` |
| checkpoint 有限性 | 未发现非有限浮点状态，optimizer step 均为 200 | 最终 checkpoint 实际读取检查 |
| checkpoint 重载评估 | 19 项统计值与保存前一致，最大绝对差为 **0** | `checkpoint_reload_eval/metrics.jsonl` |

证据相对目录统一为 `outputs/repair_20260907/`。最终真实训练平均损失为 16.5742，平均匹配查询为 422，AMP scale 全程为 32；验证 8 张图，平均查询 500，总匹配查询 token 为 4000。这里不包含 DN token，不能作为训练总计算量。

高密度测试采用 320×320 输入，峰值 CUDA 分配约 **6640 MiB（6.49 GiB）**；原始标注与有效标注的差异来自数据加载器的合法框过滤，未按数量截断标注。该压力测试不代表所有尺寸、增强与后台显存占用都不会 OOM。

四档混合合成测试使用 `[300,500,900,1500]`，有效匹配查询 token 为 3200，对比 batch-max 的 6000，理论执行 token 减少约 46.67%；完整模型分组与非分组有效输出通过 `atol=1e-5, rtol=1e-4` 一致性检查。未据此宣称真实延迟降低。

**特别说明：** 最终 8 张图的 AP 约 0.0525、AP25 约 0.3960，只是链路核对值；样本少、输入尺寸小且模型仅短训，不能用于研究结论。此版本既没有按这些指标选最佳模型，也没有生成 `checkpoint_best.pth`。

### 已建立的验证层次

- 饱和半精度、异常 count/boundary、EMA 污染、初始化、候选有效性、标签越界、DN、查询掩码、断点状态和评估口径的单元回归。
- 全部实验配置加载与 Python 语法检查。
- 真实数据有界训练，包括教师阶段和 epoch 6 纯预测路由阶段。
- 完整模型 GPU 前向/反向，确认 AQBA、边界头、密度头、DGFC、decoder 梯度有限且非零。
- 同预算及四档混合预算的分组/非分组预测一致性，容差 `atol=1e-5, rtol=1e-4`。
- 真实最密集图像的 1500 查询训练压力测试。
- 保存模型重新加载评估，并核对指标；不将短测断点伪造成完整 epoch。

## 4. 使用方式

以下命令在项目目录运行，使用专用环境。输出目录必须是新的。

### 短测，不运行完整训练

```powershell
& 'D:\venv\AQFC-DETR\Scripts\python.exe' main.py `
  --config configs/aitodv2/aqfc_runtime_smoke.py `
  --output-dir outputs/my_new_smoke `
  --max-train-steps 20 --max-eval-steps 2 --num_workers 0
```

### 正式开发训练与验证集选模（本次未执行）

```powershell
& 'D:\venv\AQFC-DETR\Scripts\python.exe' main.py `
  --config configs/aitodv2/aqfc_r50_5scale_local8gb.py `
  --output-dir outputs/my_new_train_val `
  --options train_split=train eval_split=val
```

以上仍使用 parser 的默认数据路径和旧权重 warm-start。默认配置保留 trainval/test 协议以兼容现有数据组织，但 test 模式不生成按测试 AP 选出的“最佳”断点。论文开发应使用 train/val；最终方案冻结后再按数据集协议报告 test 指标。

不建议在这台 8GB GPU 上直接切换到 800/1333 的原始大图配置；本次通过的是本地低分辨率配置。

### 本地 COCO API 重装

```powershell
.\scripts\build_ops.ps1 `
  -Python 'D:\venv\AQFC-DETR\Scripts\python.exe' `
  -OpsDirectory 'D:\PythonProject\AQFC-DETR\cocoapi-aitod\aitodpycocotools'
```

本机直接调用 pip 构建会遇到 Visual Studio 环境探测问题，已有构建脚本显式设置正确的 MSVC/SDK 环境后可成功编译。

## 5. 复审结论与尚不能宣称的结果

工程审查按数值计算、数据/查询索引一致性、异常处理、日志真实性、checkpoint 与评估定义分别进行。新增失败先复现、修复后重测；复审发现的 DN 分支等问题也纳入回归，而不是只审查第一批改动。

本次未验证多 GPU 分布式训练、完整 24 epoch、三个随机种子统计、正式分辨率的最坏显存、完整 AI-TOD/VisDrone AP，亦未证明真实推理延迟降低。四档预算的 token 数下降不等于延迟必然下降。

短测 AP 仅用于检查评估链路，不用于判断新算法优于旧模型。边界引导目标仍为固定参考统计，数量、边界及密度分支仍需充分训练。旧权重迁移覆盖率高不代表这些新增分支已经收敛。

当前修复改变了初始化、路由量定义和边界 EMA 行为；旧实验结果不可直接沿用为新版本结果。论文需在固定代码版本、相同环境和划分下重新进行完整对照实验。
