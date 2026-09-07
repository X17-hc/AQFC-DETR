# AQFC-DETR 第二轮多方面代码审查与修复报告

日期：2026-09-07。项目：`D:\PythonProject\AQFC-DETR`。

## 1. 结论与证据边界

本轮审查仍发现了真实问题，主要集中在训练状态、数据增强、评估入口、密度监督边界与配置组合，而不是缺少 Python 依赖。已修复下列确定性问题，并对同步到真实项目中的代码重新执行测试。

- **91 项自动化测试全部通过**；相较此前 72 项，本轮新增 19 项回归用例。
- **19 份主配置/消融配置校验通过，98 个活动 Python 文件 AST 解析通过**。统计排除了历史目录、输出目录及第三方 COCO API。
- 专属虚拟环境 `D:\venv\AQFC-DETR` 的 `pip check` 通过。
- 真实 AI-TOD-V2 两组各 20 步短训均完成：每组 20 次参数更新、0 次 AMP 跳步。后一组使用最终密度 mask 修复，启用增强、EMA，关闭教师路由。
- 普通与 EMA 小规模评估正常；保存、重载 EMA 后的 19 项 COCO 指标与训练内 EMA 评估完全一致。
- 最终代码在不同奇数尺寸输入上完成 CUDA 前向、反向和参数更新；DN 开启/关闭两条路径均通过。四档查询分组与 batch-max 的有效输出数值一致。
- 最密集真实图像的 1500 查询压力测试通过，峰值约 **6640.5 MiB**。

因此，当前项目已验证可以在本机进行单卡训练及评估。但这不等于已经验证完整 24 epoch 收敛、论文 AP 提升、所有数据样本与所有硬件组合，也不构成“绝无其他缺陷”的保证。

本轮没有启动完整训练，没有提交或推送 Git，没有修改原始 DQ-DETR 项目、数据集或历史实验记录。保留了既有未提交改动。

## 2. 审查方法和范围

采用“先构造失败用例—修改实现—全套回归—真实数据验证—再复审”的方式，覆盖：

1. 模型训练状态、EMA、optimizer/scaler/scheduler 与 checkpoint 恢复；
2. 原始图片/标注读取、空目标处理、Mosaic、Copy-Paste；
3. AQBA 密度监督及 encoder padding 的一致性；
4. 动态查询、DN、Matcher/损失掩码、后处理与分组推理；
5. 独立普通/EMA 评估、实验结果溯源；
6. 配置危险组合、单卡/全局日志计数；
7. 架构描述和实际实现是否一致。

第一批 12 个失败用例在修改前均可复现。后续补充状态恢复、独立评估和非整除尺寸测试。完整检测掩码等已有测试一并重跑，未仅测试新增代码。

## 3. 已发现并修复的问题

### 3.1 EMA 错误平均整数计数和状态缓冲区

位置：`util/utils.py`，以及 `main.py` 的 EMA 初始化顺序。

原实现将整个 `state_dict()` 按顺序 zip 后统一做指数平均。整数计数会先变成小数、再截断，例如计数 7 在 decay=0.9 时可能仍保存为 0；AQBA 本身已经使用 EMA 的边界状态还会被再平均一次。仅按值顺序匹配也缺少名称一致性保护。

修复：

- 通过键名对应状态，键集合不匹配时明确报错；
- 仅平均浮点/复数模型参数；计数器、运行均值、边界历史等 buffers 直接复制；
- 支持解包 DDP 模型；
- 在 DDP 初始参数广播后才构造 EMA，避免各 rank 从不同随机初值复制 EMA。

验证覆盖整数计数、浮点参数平均、运行状态复制、完整 set、真实 EMA 训练与评估。DDP 初始化顺序为代码级修复，本机未进行真实多卡运行。

### 3.2 空目标递归采样可能栈溢出

位置：`datasets/coco.py`。

原代码遇到空目标时递归调用 `__getitem__`，全空数据或极端增强序列可触发 `RecursionError`，而且注释声称不再过滤 Mosaic 空结果，实际代码仍在递归。

修复：增强前最多重试 32 次寻找非空原始图；仍失败则提示检查标注或关闭过滤。增强后合法出现的空图作为背景样本保留，不再递归。没有将读取整个数据集作为无限重试的替代方式。

这是有界保护，不保证任意稀疏数据在 32 次随机采样内必然找到目标。

### 3.3 损坏图片被静默替换，可能污染评估

位置：`datasets/coco.py::_get_raw_item`。

原实现加载失败后读下一张图并更换 image_id。验证集中的坏图可能因此被跳过，另一张图被重复评估，不能再认为评估覆盖原定集合。

修复后读取失败会报告 dataset index 和 image_id，并保留原异常链；不静默替换样本。应修复损坏数据本身，而不是让模型继续生成不可信的评估。

### 3.4 Mosaic 图像与框缩放比例不一致

位置：`datasets/mosaic.py`。

PIL resize 的宽高会分别取整，但标注原来统一乘理论 scale。例如 13×9 图像放入 5×5 格子，实际 resize 为 5×3；旧代码纵向仍使用 5/13，而非 3/9。对微小目标，亚像素级框偏移也值得纠正。

修复后横纵坐标分别使用实际 `sw/iw` 与 `sh/ih`；area 根据最终保留框重新计算；缺少 iscrowd 时补齐对应长度。新增确定性坐标/面积回归测试。

### 3.5 Copy-Paste 的低 IoU 不代表没有遮挡微小目标

位置：`datasets/copy_paste.py`。

10×10 patch 完全覆盖 1×1 目标时 IoU 只有 0.01，原 0.15 阈值会允许覆盖，但仍保留被遮挡目标的标注。

现在使用 `max(IoU, intersection/area_existing)` 判断重叠，既检查交并比，也检查原目标被覆盖的面积比例。方法更名为 `_overlap_with_existing`，避免仍称作单纯 IoU。

Mosaic 和 Copy-Paste 同时增加 masks/keypoints 输入保护：当前实现仅支持检测框，不能静默遗失分割/关键点标注。配置层也拒绝 masks 与这些增强同时启用。

### 3.6 密度监督用取整比例推算 padding，存在一格误差

位置：`models/aqfcdetr/transformer.py`、`query_allocator.py`。

此前依据图像尺寸比例 round 计算密度监督有效区域，不能总是复现 backbone 使用的 nearest-neighbor mask 下采样。例如 9→3 的网格采样位置为 0、3、6，原 4 像素有效区域实际包含两格，而 round(4/9×3)=1。

修复后 Transformer 将 level 0 的真实有效 mask 作为 `density_valid_mask` 传给辅助损失；密度目标构造和 focal loss 使用这一 mask。只在独立调用且未提供真实 mask 时保留尺寸推算兼容路径。密度图形状不匹配、完全无有效网格会报错。

验证包含专门的非整除尺寸单元测试，以及最终不同奇数尺寸的真实 CUDA 混合 batch 前反向和分组等价性测试。

### 3.7 危险配置组合原本可以通过校验

位置：`util/config_validation.py`。

新增拒绝或检查：

- `force_query_budget` 的类型及实际预算上下界，避免 100 等值被静默夹为 300；
- 非 standard two-stage、重复 two-stage pattern、额外查询、保留全部 encoder tokens、逐层改变查询数等尚未满足动态 mask 合约的选项；
- 使用学习型初始 query 内容时，其 embedding 容量不足以覆盖最大预算；
- 空间系数非有限或负值、不支持的通道门控名称；
- Mosaic/Copy-Paste 概率不合法、二者之和超过 1；
- 分割 masks 与仅支持 boxes 的增强同时启用。

数据集内部也检查实际增强概率，防止 epoch 调度后的概率被静默截断。这是明确拒绝未实现的组合，不是宣称已经实现这些旧 Transformer 选项。

### 3.8 缺失 epoch 的 checkpoint 被当成完整续训

位置：`util/checkpoint.py`。

原实现对缺失 epoch 默认使用 -1，再返回 0；如果文件含旧 optimizer 状态却缺 epoch，会出现“保留旧优化器但从 epoch 0 开始”的混合状态。

现在训练恢复必须有 epoch，且为非负整数；缺失时直接拒绝。仅模型评估不要求训练 epoch。此前对旧权重、短测断点、optimizer/scheduler/scaler/EMA/best_metrics 的保护仍保留。

### 3.9 未保存随机状态，恢复后随机流重新开始

位置：`main.py`、`util/checkpoint.py`。

新增保存验证结束时每个 rank 的 Python、NumPy、Torch CPU、CUDA 随机状态，并在训练恢复时还原；rank 数不匹配时拒绝，避免错配。旧原生 checkpoint 没有随机状态时允许兼容加载，但明确警告无法保持随机流。

**边界：这不是“位级精确续训”的承诺。** Copy-Paste 缓存、worker 内局部增强缓存没有序列化，GPU 非确定性算子也可能影响结果。本轮验证的是随机生成器的恢复和必要训练状态，不是跨进程/跨硬件的每一步完全重放。

### 3.10 独立评估无谓依赖训练集

位置：`main.py::build_data_loaders`。

独立 `--eval` 原来仍构建训练集、随机采样器与训练 loader，仅部署验证数据也可能启动失败。

现在评估只构建指定评估 split，不构建训练 loader 或训练 LR scheduler。训练 loader 为空则提前给出说明性错误。测试用模拟数据构建器断言评估不访问 train split。

### 3.11 缺少独立 EMA 评估选择

位置：`main.py`。

原 `--eval` 总是评估 `model`，即使文件名叫 `checkpoint_best_ema.pth`，也不能据文件名推断实际用了 EMA。

新增显式 `--eval-ema`，仅允许与 `--eval --resume` 一起使用；选择 `ema_model`，缺失则报错。日志记录 `evaluation_weights`。普通评估不再仅因为训练配置含 use_ema 就强制加载 EMA。

真实验证：训练内 EMA 的 19 项 COCO 指标与独立重新加载的结果最大绝对差为 0。

### 3.12 多卡步数统计混用了局部/全局口径

位置：`engine.py`、`util/runtime.py`。

原 metric 同步后 optimizer 总数是所有 rank 之和，却仍减去单 rank 的迭代次数，可能产生负的 AMP 跳步数。

现在 `train_iterations/optimizer_steps/amp_skipped_steps` 使用一致的本 rank 计数，另提供 `global_train_iterations/global_optimizer_steps`。回归用例覆盖“两卡各 10 次尝试、9 次更新”的计数关系。没有将这一模拟计数测试表述为实际多 GPU 训练验证。

### 3.13 架构说明与当前代码不一致

位置：`ARCHITECTURE.md`。

纠正了教师/预测路由计数单位、DGFC 实际先空间后通道的运算顺序，以及密度特征与单通道密度图各自进入哪个分支的描述。

特别指出：现有固定查询和纯语义排序配置仍继承 DGFC，AQBA 辅助损失也未关闭；它们是固定查询/排序对照，不是“无 AQBA、无 DGFC”的纯 DETR 基线。完整模块开关消融尚未实现，不应将这些结果直接填入原计划的完整 A–G 核心消融表。本轮没有为凑齐实验矩阵擅自替换模型算法。

## 4. 真实运行结果

环境沿用专属 AQFC-DETR 虚拟环境：Python 3.11.9，PyTorch 2.7.1+cu118，torchvision 0.22.1+cu118，NumPy 1.26.4，RTX 4060 Laptop 8GB。

| 测试 | 实际覆盖 | 结果 |
|---|---|---|
| `ema_smoke20` | 真实数据、local8gb、EMA、普通/EMA各3批评估 | 20/20更新，0跳步，平均 loss 15.5511、查询440 |
| `ema_reload` | 显式独立 EMA 加载/评估 | 19项COCO指标差值全部0 |
| `aug_teacher_off20` | 最终代码，epoch6，teacher=0，Mosaic p=.25、Copy-Paste p=.35、EMA | 20/20更新，0跳步，平均 loss 26.2960、查询500 |
| `final_odd_sizes.json` | DN开启，129×131/137×149混合训练输入，四档混合推理 | 前反向及分组等价性通过 |
| `final_no_dn.json` | DN关闭、epoch0教师路由、混合奇数尺寸 | 前反向及分组等价性通过 |
| `final_dense.json` | image_id=9736，原1729条标注、合法保留1727条，320×320、1500查询 | loss 58.7848有限，反向/更新通过，峰值6640.5MiB |

两个20步测试设置不同，loss 不能横向用来判断精度升降。后一组设置 `--start_epoch 6` 是为了立即覆盖 teacher=0 路径，不代表训练了前六个 epoch。增强通过概率启用，未把每种增强的实际触发次数作为此次统计指标。

合成分组测试采用 `atol=1e-5, rtol=1e-4`，比较有效查询 logits 和 boxes，预算为 300/500/900/1500。它验证正确性，不是实测端到端加速结论。

最密集图像测试按数据准备规则排除两条无效标注，没有为了节省显存随意截断目标集合。8GB 显存仍有边界，不能据单张压力测试承诺任意更大分辨率或 Mosaic 密集组合均不 OOM。

## 5. 结果文件与修改保护

全部本轮证据位于 `outputs/review_round2/`，主要包括：

- `final_pytest.xml`、`final_pytest.log`；
- `ema_smoke20/metrics.jsonl`、`ema_reload/metrics.jsonl`；
- `aug_teacher_off20/metrics.jsonl` 与对应 checkpoint；
- `final_odd_sizes.json`、`final_no_dn.json`、`final_dense.json`；
- 各测试对应的 `*_console.log`；
- `before/`：本轮覆盖前的文件备份；验证工具的旧版单独保存为 `before/verify_runtime.py`。

测试产生的 checkpoint 均明确带 `smoke_test=true`、`epoch_complete=false`，不得用于假装完整 epoch 的正常续训；可用于独立评估，或明确作为 warm-start 初始化。最终短测 checkpoint 已核实包含 EMA、scale=32、epoch、随机状态与短测标记。

## 6. 后续正确运行方式

保持 PyCharm 的专属解释器，不需要换回其他虚拟环境。本轮没有操作 PyCharm UI；测试通过与其相同的解释器直接调用 main.py。

新的正常训练应选择未使用的输出目录，例如：

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe main.py `
  --output-dir outputs/aitodv2_after_review_round2
```

此命令使用现有 local8gb 默认配置并启动完整训练，不是本轮已经执行的测试。输出目录若已有 checkpoint，请不要覆盖；只有完整训练 checkpoint 才用于 `--resume`。

开发阶段需要在验证集选模时，同时覆盖两个 split：

```powershell
--options train_split=train eval_split=val
```

独立评估 EMA 必须显式选择：

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe main.py `
  --eval --eval-ema `
  --resume outputs/your_full_run/checkpoint_best_ema.pth `
  --output-dir outputs/your_ema_evaluation
```

不带 `--eval-ema` 时评估普通 model；不能仅根据 checkpoint 文件名判断权重分支。

## 7. 仍需后续实验验证的事项

1. 完整训练稳定性、收敛、全量 AP/APvt/APt、三随机种子复现尚未在本轮验证。
2. 主论文分辨率与 local8gb 的图像尺度不同，小分辨率可运行不能替代原论文设置的显存和精度测试。
3. 真实多 GPU DDP、跨主机恢复、Windows 多 worker 长时间运行未实测。
4. 分组减少查询 token 不必然降低端到端延迟，需要独立 GPU 计时实验；本轮没有宣称加速百分比。
5. 完整模块开关消融尚未实现；当前配置名称对应的局部对照不能替代纯基线。
6. 已恢复随机流，但增强缓存与非确定性算子仍限制位级精确重放。
7. AQBA/DGFC 含空间卷积、归一化和池化，未验证同一图像在不同 padding/batch 组成下输出完全不变；本轮确认的是固定同一 batch 时分组与非分组 Decoder 等价。

以上限制应与已复现并修复的代码故障分开报告。当前可据实认定：**本机专属环境依赖正常，已验证单卡训练、普通/EMA评估、动态查询和必要状态恢复路径可运行；研究性能结论仍需正式实验。**
