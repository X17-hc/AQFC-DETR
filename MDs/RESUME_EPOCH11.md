# 宕机后恢复：epoch11—23

已读取桌面训练日志，并在服务器核验完整checkpoint。

- 原运行：`outputs/p2_legacy24/20260915_033533_937498_8821b83a63f3`。
- 已完整训练epoch0—10，共11轮；日志在epoch11的首批后中断。
- 恢复源：该目录的 `checkpoint0010.pth`，约571MiB，epoch=10，epoch_complete=True，7009步。
- 成功更新77067次，AMP scale=16，主学习率1e-4、骨干1e-5，scheduler.last_epoch=11。
- PyCharm选择：**AQFC-DETR 宕机续训epoch11至23及最终评估**。
- 保持GPU1、workers2、batch2、AMP、原数据及 `p2_24e_legacy.py` 全部训练参数。
- 用 `--resume` 恢复；`--no-pretrained` 清除默认初始化，不传 `--unique-output-dir`。
- 总epochs仍为24，不是新增24轮；从日志编号11恢复（自然计数第12轮），还剩13轮。
- 恢复后输出：`/workspace/AQFC-DETR/outputs/p2_legacy24/resume_epoch11_20260916`，不覆盖宕机前日志和checkpoint。
- 每轮继续保存，epoch23后完整评估。未保存的epoch11开头计算会重做，不能逐batch续接。

服务器CPU预检已实际通过模型严格加载、optimizer、scheduler、AMP、CPU随机状态和质量进度恢复，模型张量与checkpoint一致；CUDA随机状态存在，实际CUDA恢复由用户启动后的正常路径完成。本次没有训练、评估或更新权重。

使用普通“运行”，不要使用Python Console。如果菜单未刷新，重新打开项目。
该配置固定从epoch10结束处恢复；如果再次中断，请改为此次续训最新的**完整**checkpoint并核对其状态，不要再次误从epoch10重跑。
由于GPU内核非确定性及硬件/运行环境变化，不保证与未中断训练逐位一致。
