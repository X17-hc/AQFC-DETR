# H1 / D0 / D1 误差切片

只读诊断。正式权重仍是 H1 `checkpoint0023`。本机没有服务器评测目录时，用已公布全量数字；有 `class_metrics.json` 时用 `tools/summarize_size_class_ap.py`。

## 已公布全量 test（14,018）

- H1 e23：AP 32.2 / AP50 68.2 / AP75 26.3 / APvt 15.6
- D0 3e：AP 32.2 / AP50 67.7 / AP75 26.5 / APvt 16.0 / APs 37.2
- D1 3e：AP 32.4 / AP50 67.8 / AP75 26.8 / APvt 16.1 / APs 37.7

Console 未打印八类 AP。服务器输出里若有 `class_metrics.json`：

```
python -u tools/summarize_size_class_ap.py --class-metrics h1=.../class_metrics.json --class-metrics d0=... --output outputs/h1_error_slices.json
```

必看 vehicle / person 的 APvt。

## 三条可证伪假设

1. **H-cls**：下一刀动分类/匹配（EMA 或下调 o2m），不要加框损失。证伪：全量 AP50 再降且 AP75 不升，或探针 AP75≤H1−0.2pp。
2. **H-vt**：D1 组合没打到 VT。证伪：拆开 Dome 项后 APvt≥对照+0.3pp，或 vehicle/person VT 先动。
3. **H-ema-joint**：EMA / DFL 0.35 / GLU 3e 足以检验平滑与权重。证伪：该条未过 AP+0.3 且 APvt+0.3 且 AP75≥−0.2pp。

D1 组合不晋升。借鉴 Dome 的拆开探针仍允许。
