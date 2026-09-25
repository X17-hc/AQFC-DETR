# C0—S2同图同GT的VT定向诊断

PyCharm选择 **AQFC-DETR C0与S2同图同GT的VT定向诊断** 手动启动。

- 服务器AQFC-DETR解释器；工作目录`/workspace/AQFC-DETR`。
- GPU环境默认`CUDA_VISIBLE_DEVICES=0`，启动前可自行调整，不代表GPU空闲。
- 复用`outputs/scale_v1/paired_diagnosis/20260924T093854_91c0d93f/C0`，不会重新推理C0，也不会修改或复制其大文件。
- S2固定为`outputs/scale_v1/s2/20260924_115851_063163_8e60e76c05c3/checkpoint0002.pth`，校验SHA-256。
- 同一1054张图、91192个GT，保留全部类别及全部105个稀有VT GT。vehicle/person通过`official_vt/class:5`、`official_vt/class:6`重点查看，不另选困难样本。
- 只对S2执行一次800/max1333、batch1、workers0、AMP评估并导出；之后CPU配对分析。
- 无训练、无TTA、无额外NMS、无展示分数阈值过滤。
- C0旧推理关键源码、输入清单、类别、query mask、导出数量及坐标契约不符时停止；不自动补跑C0。

新输出根：`/workspace/AQFC-DETR/outputs/scale_v1/c0_s2_vt_diagnosis`，每次建立唯一子目录。

主要产物：`S2_console.log`、`S2/`预测和proposal、`paired_console.log`、`paired/`逐GT与错误分解、`targeted_summary.json`、`targeted_gt.jsonl`和`pair_manifest.json`。

**epoch0=C0，epoch1=S2**，是模型标签而非训练轮次。子集指标不是完整test AP；耗时不是严格模型测速。旧C0目录必须保留，以便追溯复用输入。

可在相同命令末尾添加`--check-only`进行CPU输入预检，不创建输出目录、不推理。本次配置准备只运行自动测试和该预检，真正诊断由用户手动执行。
