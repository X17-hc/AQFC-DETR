# F2：固定六层Encoder＋局部框细化

## 目的与证据边界

P2迁移V2的32图验收失败；memory分组替换表明恢复参考P2可基本恢复检测质量。本版本**暂不使用P2压缩**，保护成熟六层表示，单独验证已有局部细化器。不是宣称轻量学生已修复，也不承诺AP≥34/APvt≥18.5或训练提速。

结构：六层原五尺度Encoder → AQBA/DGFC → spatial候选 → 六层Decoder → P2/P3局部框细化。没有P2替代、末端融合或参考Encoder。F0/F1/P2迁移历史配置保持不变。

## 运行顺序（用户手动）

1. `AQFC-DETR F2六层细化合成与初始化检查`：合成测试与固定32图F0/F2初始化检查，不训练真实数据。部署时已执行的状态见验证报告，可按需复核。
2. `AQFC-DETR F2六层细化20步及评估`：独立输出，20个训练step＋2个评估batch。其checkpoint不能用于正式resume。
3. `AQFC-DETR F2六层细化首个完整epoch及评估`：从S2重新初始化，完整训练epoch0、完整test评估，然后停止。**配置总周期始终24，不是1。**
4. 首轮完整检查通过后，输出目录产生`first_epoch_review.json`、`resume_command.txt`、`resume_command.json`和`F2_resume_24e.run.xml`。先查看评估/AMP记录，再将XML导入或复制到本地项目`.run`，手动选择`AQFC-DETR F2六层细化续训至24轮及最终评估`。

续训从真实`checkpoint0000.pth`恢复到epoch1，完成剩余23轮，不额外再训24轮。续训保持原输出目录，不带`--unique-output-dir`或`--pretrained`。生成的配置没有启动任务的副作用。

三个初始入口均使用现有远程AQFC解释器，工作目录`/workspace/AQFC-DETR`，仅环境变量默认选择GPU2。启动前可在PyCharm调整GPU；没有代码硬绑定和占用拦截。

## 固定配方

- S2普通模型：`outputs/scale_v1/s2/20260924_115851_063163_8e60e76c05c3/checkpoint0002.pth`。
- SHA-256：`40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928`。
- 公共参数严格加载，只允许`local_refiner.*`新增；不读取S2优化器，不读取适配权重。
- AI-TODv2 trainval→test；train成员与test评价仅用于工程验证。
- 训练温和多尺度640/704/768/800、max1333，无预缩小/随机裁剪；评估800/max1333。
- batch2、workers2、seed42、AMP、无EMA；AdamW、weight decay1e-4、clip0.1、DN100。
- 公共参数LR3e-5、Backbone3e-6、新细化器1e-4；500次成功更新预热；epoch16/22分别进入0.1/0.01倍LR。
- 原训练phase固定23；quality_blend=0.25，无重新预热；geometry_v1关闭。
- 继承Mosaic/Copy-Paste及其phase23调度；实际概率和解析配置须随实验记录，不把声明值当运行值。
- 主匹配用粗框；最终回归粗/细各0.5，最终质量目标用细化框；DN和其他辅助层规则不变。
- 每轮保存，只在epoch0和epoch23完整test评估，不按test选最佳轮次。

## 停止、断电与恢复

`--stop-after-epochs N`是本进程完成的epoch数，默认0；不进入结构签名或改变总LR日程。与训练/评估step限制、debug和eval-only组合报错。

F2每个安排评估的完整epoch在评估前先保存`checkpointNNNN_pending_eval.pth`。此文件保存完整训练边界和RNG，`evaluation_state=pending`。它不是已验收终点，专用续训入口不会把它当作首轮成功。

- 评估完成后，原子保存`checkpoint.pth`及该轮编号checkpoint，标记`evaluation_state=complete`。
- 评估中断时，不会生成首轮通过或续训配置。保留pending文件和日志，请提供路径进行独立评估恢复；不要重命名pending文件来绕过状态检查。
- 后续非评估轮也原子保存完整编号checkpoint，状态`not_scheduled`。
- 从最新完整epoch恢复，不支持精确batch游标恢复；中途训练断电可能重做当前未完成epoch。
- 临时保存失败不会覆盖上一份完整checkpoint；需满足实际可写空间，未预分配24轮权重空间。

## 首轮判定

要求7009个完整训练batch、有效参数更新/AMP计数一致，且legacy评估覆盖14018个唯一test ID。计数变化需要排查，不能静默接受不同划分。

S2历史完整test：AP31.8787、APvt15.6964、AP75 25.3104。首轮AP或APvt下降超过1个百分点，或完整性检查失败，生成`PAUSE_AND_REVIEW`，不产生续训XML。该门槛只是工程排查信号，不是显著性检验或自动选模。

通过时状态为`READY_FOR_MANUAL_REVIEW`，不是保证24轮必然涨点。日志记录每轮成功更新和AMP跳步；训练/评估分别计时，总时间还包括保存I/O。stdout/stderr在`launches/<唯一ID>/console.log`完整保留，不受PyCharm显示缓存截断影响。

## 对照和最终结论

没有自动增加F0完整24轮。F2相对S2的变化包含新增训练和细化的共同影响，不能全部归因于细化模块。本版本若提升精度但变慢，必须如实报告。最终目标要求同一普通checkpoint在完整test同时达到AP34/APvt18.5；test已参与设计，达到后仍先称工程目标达成。

## 结果记录模板

| 项目 | 首轮 | epoch23 |
|---|---|---|
| checkpoint路径/SHA | 待手动运行 | 待手动运行 |
| 完整训练步数/成功更新/AMP跳步 | 待填 | 待填 |
| 完整评价图像/唯一ID | 待填 | 待填 |
| AP/APvt/AP75 | 待填 | 待填 |
| 训练/评估秒数 | 待填 | 待填 |
| GPU UUID及资源竞争 | 待填 | 待填 |
| 粗框→细化框表现、异常说明 | 待填 | 待填 |

首轮及完整24轮真实训练均待用户执行，不用单元测试代替训练验收。
