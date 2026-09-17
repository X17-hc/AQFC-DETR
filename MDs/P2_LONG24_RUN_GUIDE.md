# P2质量对齐：新增24轮训练及最终评估

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent（实验来源与边界记录）。
- 模式：配置实施；2026-09-15。
- 状态：仅准备手动入口，不自动启动训练；24轮性能与精度尚未验证。

## 在PyCharm中启动

选择 **AQFC-DETR P2质量对齐24轮训练及最终评估**，使用普通“运行”，不要选择“在Python控制台中运行”。
本地XML：`.run/AQFC-DETR_P2_24e.run.xml`；模型配置：`configs/incremental_v2/p2_24e.py`。
如菜单未刷新，重新打开AQFC-DETR项目。远程工作目录为 `/workspace/AQFC-DETR`，解释器为 `/opt/conda/envs/AQFC-DETR/bin/python`。

| 项目 | 设置 |
|---|---|
| 初始化 | 同一epoch10普通模型权重，SHA-256锁定；不加载旧optimizer |
| 新训练长度 | 24个完整epoch，本轮编号0—23 |
| 模型/增强阶段 | 11—34，总阶段长度35；不重启教师课程 |
| 数据 | AI-TODv2 trainval → 完整test；工程实验，不是未见val研究对照 |
| 设备 | GPU3，仅通过运行配置环境变量设置，无代码绑定 |
| batch / workers / AMP | 2 / 2 / 开启；EMA关闭 |
| 主学习率 | 新增第1—16轮1e-5，第17—22轮1e-6，第23—24轮1e-7 |
| 骨干学习率 | 主学习率的1/10 |
| 质量监督 | P2 quality_blend，λ最大0.25，首轮按成功更新数预热 |
| 算法 | light_dw、vectorized、spatial；启用已验证的工程优化 |
| 保存与评估 | 每轮保存；只在最后一轮进行完整评估；不按test挑选最佳轮 |
| 输出 | `/workspace/AQFC-DETR/outputs/p2_long24/` 下自动创建唯一子目录 |
| 进度 | 训练每1000batch，评估每2000batch；保留首末批和结束汇总 |

初始化文件：
`/workspace/AQFC-DETR/outputs/server_update24/20260910_235559_501337/checkpoint0010.pth`

SHA-256：`898f7542b4c7b96317e857d9df5708ed1de63a8644f9edfaed3f080b2e7e272a`。
运行入口会严格核验epoch、完整性、结构和全部模型state覆盖，失败直接停止，不换用其他权重。

## 与旧实验的区别

这是从已训练模型**重新建立optimizer的24轮长周期微调**，不是从零训练，也不是从旧P2三轮终点续训。
扩大总阶段长度到35会改变allocator权重与Mosaic的后期衰减轨迹；这是明确的新配方，不能将长训练收益全部归因于质量loss或空间选择优化。
主学习率采用保守低值与后期衰减，尚未验证它是最佳长期训练配方。
原三轮P1/P2和原服务器24轮配置保持不变。

## 启动前及运行后

1. 手动确认GPU3适合运行；不与其他训练/benchmark并发测速，不终止他人任务。
2. 确认原数据、初始化权重可读，并为24个完整checkpoint及日志预留存储；不删除旧结果。
3. 训练启动后核对配置为24轮、workers2，迁移为完整模型覆盖，首轮阶段为11。
4. 每轮应有7009个训练batch（当前14018张trainval、batch2），检查成功更新与AMP跳步，不仅看loss。
5. 最后完整评估应覆盖14018张test图像；普通模型评估，800尺度、max_size=1333，无TTA。
6. 大致预算38—42小时，基于历史完整epoch约1.5小时加最终评估；资源竞争及增强会改变耗时，不能保证AP持续上升。

本配置没有自动监控、启动、停止或续训任务。异常时保留现场，不自动重试/降尺度。
中断后不要直接再次点此入口来“续训”：它会新建目录、重新从epoch10开始。
真正续训需要使用该次输出checkpoint、移除 `--pretrained` 和 `--unique-output-dir`、指定原输出目录，并保持workers/batch/数据成员/配置不变；加载前还应核对checkpoint的 `criterion_progress` 和完整epoch状态，避免把缺失进度当作0。未在本次交付中声称已完成24轮中断恢复验收。
