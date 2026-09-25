# S1原尺度细节保留：手动运行

先选择 **AQFC-DETR S1原尺度细节保留20步及评估**，成功后单独选择
**AQFC-DETR S1原尺度细节保留3轮及最终评估**。交付过程不启动真实训练。

- 远程解释器：`/opt/conda/envs/AQFC-DETR/bin/python`。
- 工作目录：`/workspace/AQFC-DETR`。GPU环境默认2，启动前自行检查并调整。
- 两项从原epoch23普通模型warm-start；不从冒烟或C0/C1/C2终点续训。
- 输出分别为`outputs/scale_v1/s1_smoke`、`outputs/scale_v1/s1`下的唯一子目录。
- 正式三轮每轮保存，最后一轮完整test评估；无自动串行任务。
- 继承C0：固定phase23、质量0.25、几何0、batch2、workers2、AMP、EMA关闭。

`native800`仅替换训练的合成增强后变换：水平翻转、等比例短边800/最长1333、原归一化。
最大边受限时短边可以小于800，并不是保持原始像素尺寸。Mosaic/Copy-Paste和调度不变，
因此不意味着完全无缩小。严格限长resize使用显式宽高，避免原resize取整偶尔产生1334像素；
旧训练与评估resize保持不变。

`legacy`为默认，签名省略该中性默认值以兼容历史checkpoint；native800记入签名，
训练resume不允许跨模式。原epoch23SHA-256检查保留。

这是trainval→test工程配方筛查。不能提前声称AP、APvt或速度提升。
当前配置仅比C0新增训练变换字段；历史C0运行源码没有完整冻结证明时，不能将其视为已确认严格对照。
先核对历史源码、初始化和数据协议，不自动补跑C0。
