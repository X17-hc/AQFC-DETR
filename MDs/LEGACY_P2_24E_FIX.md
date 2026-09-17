# 旧论文权重初始化P2：24轮配置修复

## 原因

`p2_24e.py` 是从特定AQFC epoch10权重进行24轮低学习率微调的配置，要求哈希、epoch、完整训练元数据和结构均一致。
`dqdetr_best305.pth` 是只有 `model` 的旧格式文件，不具有上述元数据，也不含当前AQBA的多数参数。
原报错是初始化来源不匹配，不是数据路径、GPU或权重损坏。不能伪造epoch或只替换哈希来通过严格恢复。

## 现在运行哪个配置

PyCharm仍选择 **AQFC-DETR P2质量对齐24轮训练及最终评估**。
它现已指向 `configs/incremental_v2/p2_24e_legacy.py`，并保留用户选择的GPU1。

- 旧检测权重：`/workspace/AQFC-DETR/weights/legacy/dqdetr_best305.pth`。
- 固定SHA-256：`f854eec1f0d0ebf6e131d01e0b2826ce5690b593a5ed7f3fb8203453a62936e4`。
- 使用部分参数迁移，不恢复optimizer、scheduler或历史epoch；仍进行哈希校验。
- 新训练epoch0—23，模型/增强课程0—23，总阶段24；新AQBA需要教师课程和辅助损失预热。
- 主学习率1e-4，骨干1e-5；完成第13、23轮后各衰减10倍，沿用已有服务器24轮配方。
- P2质量混合监督、工程优化、batch2、workers2、AMP保留；EMA关闭。
- 每轮checkpoint，最后一次完整test评估；trainval→test仅作工程实验，不是从零训练或无泄漏的train→val研究实验。
- 新输出根目录：`/workspace/AQFC-DETR/outputs/p2_legacy24`，启动时创建唯一子目录。
- 不修改原 `p2_24e.py` 的epoch10保护；选择不同初始化必须同时选择对应配置。

手动命令的关键变化：

```text
--config /workspace/AQFC-DETR/configs/incremental_v2/p2_24e_legacy.py
--pretrained /workspace/AQFC-DETR/weights/legacy/dqdetr_best305.pth
--output-dir /workspace/AQFC-DETR/outputs/p2_legacy24
```

原 `P2_LONG24_RUN_GUIDE.md` 描述的是epoch10微调配方，其学习率、阶段范围和初始化不能用于当前旧权重入口。

## 验证与限制

本地真实模型CPU迁移：state-dict元素覆盖98.9453%，仅参数覆盖98.9123%，形状不匹配0项。
主干、Encoder、Decoder、输入投影、分类/回归头完整加载；AQBA按state元素仅覆盖0.189%，其余保持新模型初始化。
所有模型参数有限；总体覆盖率不是功能等价或初始AP保证。
完整迁移报告会在真实启动后的输出目录保存为 `checkpoint_migration_report.json`。

10项本地专项测试通过，覆盖身份检查、部分迁移、旧checkpoint禁止resume、新旧课程隔离及PyCharm解析。
只做CPU模型构建、加载与测试；没有启动真实数据训练或完整评估。24轮是否正常完成及最终AP仍待手动验证。
未删除旧日志、输出或权重，也未修改其他项目。
