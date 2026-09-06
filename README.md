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

建议 Python 3.9、PyTorch 2.4、CUDA 12.4。安装依赖后编译多尺度可变形注意力：

```bash
pip install -r requirements.txt
bash scripts/build_ops.sh
```

Windows 需使用与 PyTorch/CUDA 匹配的 Visual Studio C++ 工具链，在 `models/aqfcdetr/ops` 中执行 `python setup.py build install`。

## 配置校验

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
