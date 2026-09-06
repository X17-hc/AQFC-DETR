# 重构验证报告

验证日期：2026-08-29。

## 已通过

- CPU 单元测试：22 passed；
- 配置：AI-TOD-V2、VisDrone 及全部消融配置通过严格校验；
- Python 静态编译：通过；
- 活动源码旧名称审计：零命中；
- 目标目录与暂存目录文件数一致；
- 两个历史权重均可识别为旧格式；
- 两个复制权重与来源文件 SHA-256 完全一致；
- 目标工程不含 `data`、`origin`、旧 `.git`、`.idea`、`__pycache__` 或 pytest cache；
- 原工程 Git 仍保持重构前已有的 dirty 状态，重构过程未向原目录写文件。

单元测试覆盖 AQBA 档位和教师调度、非有限回退、EMA eval 稳定性、密度 Gaussian 与 padding mask、密度 focal 反向、五种 DGFC gate、proposal fused/mixed 去重、查询 mask、Matcher padding 排除、checkpoint 映射和分组恢复顺序。

## 环境阻塞项

当前 Windows Python 环境没有编译 `MultiScaleDeformableAttention`，因此依赖该扩展的完整模型导入测试被 pytest 跳过 1 项。当前环境也没有 AI-TOD-V2/VisDrone 数据路径和可用 CUDA 训练环境，故未执行：

- CUDA 扩展编译验证；
- 完整模型合成前向/反向；
- 20 iteration 真实数据冒烟；
- 50 次预热、200 次计时的 GPU 效率基准；
- 多日论文训练与 AP 验收。

这些属于环境/数据验证，不是静态重构的伪通过项。准备好匹配的 PyTorch、CUDA、C++ 编译器和数据后，依次运行：

```powershell
cd D:\PythonProject\AQFC-DETR\models\aqfcdetr\ops
python setup.py build install
python test.py

cd D:\PythonProject\AQFC-DETR
python -m pytest -q tests
python tools\benchmark_dynamic_queries.py --config configs\aitodv2\aqfc_r50_5scale_24e.py --checkpoint <new-checkpoint> --batch-size 1
```

旧权重的实际模型加载覆盖率报告会在首次使用 `--pretrained` 构建完整模型后生成到输出目录；由于当前扩展未编译，本次没有伪造覆盖率数字。
