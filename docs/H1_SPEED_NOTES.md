# H1 DataLoader 默认值与评测税

`main.py` 全局默认仍是 `num_workers=0`、`persistent_workers=False`（本机 Windows/PyCharm 安全）。**在服务器矩阵写回数字之前不要改这个默认。**

服务器 H1 训练入口继续显式 `--num_workers 2`。新增 `--persistent-workers`，仅在 `num_workers>0` 时生效。

## 入口（人在服务器启动，计划不串训）

1. `AQFC-DETR H1 workers吞吐矩阵`：打印 0/2/4 × persistent 命令。按 `tools/benchmark_h1_workers.py` 跑 50 步（H1 smoke，Mosaic 开着）。吞吐下降 >5% 不算提速成功。
2. `AQFC-DETR H1配方含Mosaic profile`：50 步 Chrome trace + 250 步（50 warmup + 200 measure）命令。禁止用 `mosaic_p=0` 短测当 epoch 时间。
3. 已有 `AQFC-DETR原主体-H1同卡短性能对照`：H1 interleaved 50+200。

矩阵写完后：只改**服务器启动器**的 `--num_workers`；Windows 本地保持 0。

## 评测税

`--eval-boxes-only` 跳过评测期 SetCriterion，前向与导出框不变。默认关。32/1000 预检和 3e 探针评测打开。对照同一 H1 权重：`detection_fingerprint` 或 `coco_places_agree`（COCO 三位小数）。

`--eval-ema-only`：EMA 探针只评 EMA 权重。
