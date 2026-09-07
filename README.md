# AQFC-DETR

AQFC-DETR（Adaptive Query and Feature Calibration DETR）面向密集微小目标检测，在 two-stage Deformable DETR 主线上加入三个协同机制：

1. **AQBA**（Adaptive Query Budget Allocator）按图像预测 300/500/900/1500 查询预算；
2. **DGFC**（Density-Guided Feature Calibrator）用低扰动通道门控和空间残差校准 encoder memory；
3. **语义—密度候选融合**以 encoder 语义 proposal 为主体，引入受监督密度先验重排候选。

推理时 backbone、encoder、AQBA 和 DGFC 对整个 batch 执行；Decoder 按实际预算分组执行，并恢复原样本顺序。训练保留 DN 查询，采用 batch 最大长度与 `query_valid_mask`，padding 查询不进入 Matcher、检测损失和后处理。

## 目录

- `models/aqfcdetr/`：模型、Transformer、AQBA、DGFC、Matcher 和 CUDA 算子；
- `configs/`：AI-TOD-V2、VisDrone 和消融配置；
- `util/checkpoint_migration.py`：旧权重 warm-start 迁移；
- `tools/`：配置校验、权重转换、效率统计、TTA 与可视化入口；
- `tests/`：CPU 单元测试及需 CUDA 扩展的集成测试入口；
- `legacy_artifacts/`：原始日志、评估结果及一次性历史脚本，只作追溯；
- `weights/legacy/`：两个原始权重及 SHA-256 清单。

## 环境与算子

2026-09-06 已在全新独立环境 `D:\venv\AQFC-DETR` 验证：Python 3.11.9、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118、CUDA Toolkit 11.8、RTX 4060 Laptop 8GB。

本机安装、启动、续训及显存限制详见 [TRAINING_WINDOWS.md](TRAINING_WINDOWS.md)，实际验证结果见 [TEST_REPORT.md](TEST_REPORT.md)。环境已配置好，不需要重新安装。

在其他兼容环境中，先安装匹配 CUDA 的 PyTorch，再安装项目依赖和算子：

```bash
pip install -r requirements-runtime.txt
pip install --no-build-isolation -r requirements.txt
bash scripts/build_ops.sh
```

Windows 使用 `scripts/build_ops.ps1` 配置 MSVC/CUDA 并编译。`scripts/setup_environment.ps1` 只创建新环境，若目标已存在会停止，不覆盖、不复用。

本机启动（完整 24 epoch，不限制训练步数）：

```powershell
cd D:\PythonProject\AQFC-DETR
.\scripts\train_aitodv2.ps1 -DataRoot D:\PythonProject\DQ-DETR\DQ-DETR\data\path\AITODv2 `
  -Pretrained weights\legacy\dqdetr_best305.pth
```

此脚本默认使用完整模型的 `local8gb` 低显存配置（batch 1、较小输入、无 Mosaic/Copy-Paste、无 EMA）。原 `24e` 论文配置未降低分辨率；两者不能直接混为同一组精度实验。

## 配置校验

### 无参数直接启动

本机可使用新环境直接运行 `main.py`，不再需要在 PyCharm 中填写脚本参数：

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe D:\PythonProject\AQFC-DETR\main.py
```

默认值直接写在 `main.py` 的 `get_args_parser()` 中，各项使用 `parser.add_argument(..., default=...)` 并附中文注释：local8gb 配置、现有 AI-TOD-V2 数据目录、旧最佳权重初始化、输出 `outputs/aitodv2_pycharm`、workers=0。默认项目内路径相对于 `main.py` 所在目录解析。此命令会启动完整 24 epoch，而不是冒烟测试。

命令行仍可覆盖默认值；`--resume` 会取消默认旧权重初始化，`--no-pretrained` 可禁用默认 warm-start。显式同时传入 `--resume` 和非空 `--pretrained` 仍会报冲突。默认预训练只表示检测模型初始化，不会关闭 backbone 原有的 ImageNet 初始化行为。

已有的 PyCharm `AQFC-DETR训练` 配置已清空重复参数。如果输出目录中已有实验记录，新实验请另设 `--output-dir`，续训请显式指定 `--resume`。

```powershell
python tools/validate_config.py --config configs/aitodv2/aqfc_r50_5scale_24e.py
```

旧配置字段会直接报错并给出新字段，不会静默回退。

## 训练与评估

```powershell
python main.py `
  --config configs/aitodv2/aqfc_r50_5scale_24e.py `
  --data-root D:\Datasets\AITODv2 `
  --output-dir outputs\aitodv2_main
```

旧权重只用于初始化：

```powershell
python main.py `
  --config configs/aitodv2/aqfc_r50_5scale_24e.py `
  --data-root D:\Datasets\AITODv2 `
  --pretrained weights\legacy\pretrain_model.pth `
  --output-dir outputs\aitodv2_warmstart
```

新格式 checkpoint 可严格恢复训练或评估：

```powershell
python main.py --config configs/aitodv2/aqfc_r50_5scale_24e.py `
  --data-root D:\Datasets\AITODv2 --output-dir outputs\aitodv2_main `
  --resume outputs\aitodv2_main\checkpoint.pth

python main.py --config configs/aitodv2/aqfc_r50_5scale_24e.py `
  --data-root D:\Datasets\AITODv2 --output-dir outputs\aitodv2_eval `
  --resume outputs\aitodv2_main\checkpoint_best.pth --eval
```

## 验证

```powershell
python -m pytest -q tests
python -m compileall -q .
```

完整模型前向/反向依赖已编译的 CUDA 扩展和实际数据。默认主方法是 fused proposal、密度权重 0.25、Tanh+空间 DGFC、分组 Decoder。论文应同时报告 AP、平均实际查询数、Decoder query token 和真实延迟。

迁移细节见 [MIGRATION.md](MIGRATION.md)，模型数据流与损失定义见 [ARCHITECTURE.md](ARCHITECTURE.md)。
# 2026-09-07 运行修复说明

第二轮多方面审查、91项测试及真实短训证据见 [第二轮审查报告](REVIEW_ROUND2_20260907.md)；
上一轮记录保留于 [修复与复审报告](REPAIR_REVIEW_20260907.md)。
新训练必须使用未包含 checkpoint 的输出目录；短测断点不能按完整 epoch 续训。
默认 trainval/test 协议仍可运行，但测试集不再用于选择最佳 checkpoint；开发选模请同时设置
`--options train_split=train eval_split=val`。

独立评估 EMA 使用 `--eval --eval-ema --resume <checkpoint>`，不带 `--eval-ema` 时始终评估普通模型。
新 checkpoint 保存各 rank 的随机状态；旧 checkpoint 缺少该状态会警告。增强缓存没有序列化，
因此不承诺位级精确续训。EMA 仅平均参数，计数和运行 buffers 直接复制。
