# Windows 本机训练说明

更新：2026-09-06。项目已经在新建环境中完成 GPU 训练验证；不是仅通过 import。

## 1. 已配置的独立环境

- 环境名称：`AQFC-DETR`。
- 环境目录：`D:\venv\AQFC-DETR`。
- Python：`D:\venv\AQFC-DETR\Scripts\python.exe`，3.11.9。
- PyTorch / torchvision：2.7.1+cu118 / 0.22.1+cu118。
- NumPy：1.26.4；CUDA Toolkit：11.8；MSVC：14.29。
- GPU：NVIDIA GeForce RTX 4060 Laptop，8GB。
- `include-system-site-packages = false`，不继承系统或其他虚拟环境的 site-packages。
- CUDA 注意力扩展及项目自带 AI-TOD mask 扩展均已编译、独立安装到新环境。

本次网络下载大包受阻时使用了本地离线安装材料，安装到新环境后发现三处 PyTorch 源文件存在历史修改，已按 PyTorch v2.7.1 及其对应 pybind11 提交恢复官方源码。其余 torch 文件已与安装材料的原始 RECORD 核对。新环境运行不需要旧虚拟环境路径；**不建议将未经核对的旧离线 wheel 当作以后重装的官方来源**。新建脚本使用官方 PyTorch 下载源。

PyCharm / VS Code 中请选择上述新 Python 解释器，不要选择系统 `D:\python\python.exe` 或其他项目的解释器。无需激活环境也可用绝对路径运行：

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. 本机推荐的正式启动命令

```powershell
cd D:\PythonProject\AQFC-DETR
.\scripts\train_aitodv2.ps1 `
  -DataRoot D:\PythonProject\DQ-DETR\DQ-DETR\data\path\AITODv2 `
  -Pretrained weights\legacy\dqdetr_best305.pth `
  -OutputDir outputs\aitodv2_local8gb
```

脚本固定使用新环境，默认 `num_workers=0`、batch size 1。此命令**会运行完整 24 epoch**，没有短训练的步数限制。本次修复没有替你启动它。

等价直接命令：

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe main.py `
  --config configs/aitodv2/aqfc_r50_5scale_local8gb.py `
  --data-root D:\PythonProject\DQ-DETR\DQ-DETR\data\path\AITODv2 `
  --pretrained weights/legacy/dqdetr_best305.pth `
  --output-dir outputs/aitodv2_local8gb --num_workers 0
```

数据集没有复制；仅从原有数据目录读取。当前训练入口沿用 AI-TOD-V2 的 trainval 训练、test 评估协议。进行超参数选择时，应另行规划训练/验证划分，不能将反复 test 调参当作独立测试结果。

## 3. 8GB 配置与论文配置的区别

`aqfc_r50_5scale_local8gb.py` 保留 R50、5 个特征层、6 层 encoder、6 层 decoder、AQBA、DGFC、密度融合、DN 和四档查询，**没有缩减模型结构**。资源调整为：

- batch size：1；
- 输入短边：256/288/320，长边上限 384；
- 关闭 EMA、Mosaic、Copy-Paste，以限制额外显存和目标数量；
- 其余训练日程继承 24 epoch 主配置。

真实最密集样本测试：1729 原始标注，经数据加载器现有合法性筛选保留 1727 个框；320×320 输入、1500 查询、DN 开启，前向/反向/优化更新通过，峰值分配显存 6637 MiB。没有为通过测试额外裁剪密集样本或截断标注。

上述测试不是所有输入/增强组合的显存上界保证。训练时关闭其他占用 GPU 的程序；若仍出现 OOM，可先将短边固定为 256，但不能将小分辨率精度与原论文设置直接对比。完整 `aqfc_r50_5scale_24e.py` 仍保留 480–800 短边和最高 1333 长边、原增强策略；本机未验证其全密度训练显存可承受性。

## 4. 续训与评估

续训必须使用新项目生成的 checkpoint：

```powershell
.\scripts\train_aitodv2.ps1 `
  -DataRoot D:\PythonProject\DQ-DETR\DQ-DETR\data\path\AITODv2 `
  -Resume outputs\aitodv2_local8gb\checkpoint.pth `
  -OutputDir outputs\aitodv2_local8gb
```

恢复模型、optimizer、scheduler、epoch，以及存在时的 EMA 和 AMP scaler。旧原始权重只能用 `-Pretrained` / `--pretrained`；不要与 resume 同时指定。

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe main.py `
  --config configs/aitodv2/aqfc_r50_5scale_local8gb.py `
  --data-root D:\PythonProject\DQ-DETR\DQ-DETR\data\path\AITODv2 `
  --resume outputs/aitodv2_local8gb/checkpoint_best.pth `
  --output-dir outputs/aitodv2_local8gb_eval --num_workers 0 --eval
```

`checkpoint.pth` 是最后一个已保存 epoch；`checkpoint_best.pth` 是已评估 epoch 中的最佳常规模型；启用 EMA 时另保存 `checkpoint_best_ema.pth`，其 `ema_model` 字段保存 EMA 权重。普通 `--eval` 评估 `model` 字段，并不自动改用 EMA 权重。

## 5. 快速自检与短训练

```powershell
& D:\venv\AQFC-DETR\Scripts\python.exe -m pip check
& D:\venv\AQFC-DETR\Scripts\python.exe -m pytest tests -q
& D:\venv\AQFC-DETR\Scripts\python.exe tools/verify_runtime.py --output outputs/runtime_verify.json
& D:\venv\AQFC-DETR\Scripts\python.exe main.py `
  --config configs/aitodv2/aqfc_runtime_smoke.py `
  --data-root D:\PythonProject\DQ-DETR\DQ-DETR\data\path\AITODv2 `
  --pretrained weights/legacy/dqdetr_best305.pth `
  --output-dir outputs/my_smoke --num_workers 0 --max-train-steps 20 --max-eval-steps 2
```

smoke 配置只有 1 epoch，使用固定小输入和非零 allocator loss 权重。短训练的 checkpoint **仅是验证产物，不是收敛权重**。2 张图片的 AP 也不是论文结果。

AMP 默认开启，初始较大的梯度缩放可能导致前几步被安全跳过。日志记录 `train_iterations`、`optimizer_steps`、`amp_skipped_steps`；scaler 不再每个 epoch 重置，并随 checkpoint 保存。若持续跳步或 loss 非有限，需要进一步诊断，不应忽略。

## 6. 如以后需要从零安装

本机环境已经存在，**不要再次执行创建命令**。在新机器或确认未存在的环境目录执行：

```powershell
.\scripts\setup_environment.ps1 -BasePython D:\python\python.exe -EnvironmentPath D:\venv\AQFC-DETR
```

目标存在时脚本立即报错，绝不覆盖/复用。该脚本需要网络、Git、CUDA 11.8 和 MSVC/Windows SDK；`build_ops.ps1` 中的工具链路径是本机实测路径，换机器需按实际安装位置调整。安装过程包含官方 torch/torchvision、运行依赖、panoptic API、项目 AI-TOD API 和 CUDA 注意力扩展。

只需重编译注意力扩展时：

```powershell
.\scripts\build_ops.ps1 -Python D:\venv\AQFC-DETR\Scripts\python.exe
```

编译脚本临时设置工具链环境，结束后恢复调用者环境。无法在受限/特殊运行宿主直接调用 NVCC 时，可在普通 PowerShell 中执行；也可使用 `-WheelOnly -WheelDirectory <目录> -OpsDirectory <源码目录>` 构建 wheel，再用新环境 pip 安装。不要把 Ninja 编译命令改成 `--version` 来绕过错误。

## 7. 验证材料

详见 `TEST_REPORT.md` 和 `outputs/runtime_verification/`。没有运行完整论文训练、全量 AP 评估或三随机种子实验，不能据此承诺模型收敛精度或 AP 提升。
