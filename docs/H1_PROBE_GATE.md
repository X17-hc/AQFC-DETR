# H1 探针人工门

对照写死：H1 e23 全量 AP 32.2 / APvt 15.6 / AP75 26.3；D0 全量 32.2 / 16.0 / 26.5。晋升：相对写死对照 AP≥+0.3pp 且 APvt≥+0.3pp，且 AP75≥−0.2pp。1000 图不当晋升。32 图不算总 AP、不算 APvt。

顺序：改代码 → `AQFC-DETR H1候选预检32图`（FP32/AMP 导出不变）→ `AQFC-DETR H1候选预检1000图`（相对 `docs/h1_precheck_baseline.json` 或改代码前同权重同 1000 ID：AP/APvt 掉 >1pp，或 VT cover@0.5 掉 >2pp → 停）→ 只开一条 3 轮。未预检不得开训。

| 入口 | 配置 | 对照 | 评测 |
| --- | --- | --- | --- |
| AQFC-DETR H1探针EMA3轮 | h1_ema_3e.py | H1；只评 EMA | `--eval-boxes-only --eval-ema-only` |
| AQFC-DETR H1探针JointDFL3轮 | h1_joint_dfl_035_3e.py | H1；只动 DFL 0.35 | `--eval-boxes-only` |
| AQFC-DETR H1探针GLU3轮 | h1_glu_3e.py | H1；禁 Swish | `--eval-boxes-only` |

三条全灭 → 不自动 24 轮；回到误差切片换假设。过门一条 → 只晋升该条，再考虑**同一配方** 6–12e。默认不加长已跑过的 D1 组合，默认不开 F2 24e。

## 探针结束后勾一项

- [ ] 晋升该条（全量过门：AP+0.3 且 APvt+0.3，AP75≥−0.2pp）
- [ ] 同配方加长 6–12e（仍不用 test 选 best）
- [ ] 停手；继续导出 H1 `checkpoint0023`
- [ ] 不开 F2 24e（默认）

不改 H1/D0/D1 已跑目录里的权重。geometry 权重保持 0。Best-checkpoint 在 test 上保持关闭。
