# H1终点微调：密度低估保护与候选去冗余

## Material Passport

- 技能：academic-research-suite / experiment-agent。
- 模式：实现与工程验证；真实实验由用户手动启动。
- 版本：protected_density_v1，2026-09-28。
- 本文不证明AP/APvt改善，也不宣称实现训练提速。

## 不变与改变

保持H1全部参数结构、六层五尺度Encoder、AQBA/DGFC、细节桥接、候选辅助头和分布式细化器。
原H1配置及初始化保护不变。新增 `dome_transfer_recipe=d0/d1` 为独立微调契约。

共同初始化：`outputs/legacy_joint/h1_epoch0/20260927_012928_172150_7b068fc7650e/checkpoint0023.pth`。
SHA-256：`15862ce9658c871c0152c5120f1562675a5ce294bf7029a4b7ed28b4bbbd9aee`。
只加载普通model，全部键严格对应，不重置辅助头，不恢复旧optimizer。

| 设置 | D0 | D1 |
|---|---|---|
| 候选选择 | 原spatial | protected_density |
| 密度低估辅助项峰值 | 0 | 0.10 |
| 新训练 | 3轮 | 3轮 |

两者其余配方相同：640/704/768/800，max1333，batch2，workers2，AMP，seed42，EMA关闭。
非骨干及成熟的桥接/辅助头/细化器LR均1e-5，骨干和低LR投影组1e-6。
前500次成功更新从10%峰值预热，此后恒定。原模型phase固定23。
quality=0.25、最终粗细框回归混合=0.5、候选辅助与DFL权重保持成熟值，不随新LR预热而重置。
每轮保存，只有末轮完整test评价；测试集结果不用于选择最佳轮次。

## 损失与选择实现

`L_under = mean_b(sum(V*y*relu(y-p)^2) / clamp_min(sum(V*y),1))`。
FP32计算，目标与mask不求梯度；空GT该项为0。该项独立加入总损失，不进入allocator总损失，避免二次加权。
新辅助项随本次成功更新数从0增长到0.10，AMP跳步不推进；原criterion进度字段保存这一计数并严格恢复。

候选保留语义前ceil(0.75K)，池为语义前max(1500,2K)（裁剪合法数量）与旧spatial选择并集。
使用Encoder回归后的归一化cxcywh框，不把anchor合法性张量误当回归框。
256分块比较更高语义排名、同合法类别候选，IoU阈值0.4+0.5*max(d_i,d_j)。
冗余只改变格内优先级，不删除核心、不永久丢弃，也不是输出NMS。
原8×8密度配额分配；格内非冗余优先、各层级按语义排序，最后按全局语义补齐。
最终统一语义排序，同分按token编号；mask、DN和所有gather共用索引。
第9通道不参与同类抑制，合法类别索引明确为0—7。
合法位置非有限输入报错；全零密度回退语义Top-K并记录。

## 手动运行顺序

1. `AQFC-DETR Dome借鉴版合成与兼容检查`：CPU测试及两组真实权重严格加载检查，不读取训练图像。
2. `AQFC-DETR Dome借鉴版20步及小规模评估`：独立1个局部epoch、20次训练迭代、2个评估batch。不能用其权重开始正式训练。
3. `AQFC-DETR Dome借鉴版32图与1000图候选预检`：32图分别检验FP32/AMP导出前后预测不变，然后同源权重跑1000图两种候选。无GT前向输入，无训练。
4. 预检通过后手动运行D0与D1三轮入口；二者必须从共同H1终点重新初始化。
5. `AQFC-DETR D0-D1同卡短性能对照`：ABBA、50预热+200测量，固定800，动态预算，稳定损失系数；不保存短测权重。

默认GPU2仅在PyCharm环境变量配置，不代表空闲。启动前可修改，不终止其他用户任务。
运行器保存完整console.log与唯一输出目录。正式与冒烟目录隔离，不自动串行启动训练。
训练日志1000batch、评估2000batch；以磁盘日志为准，PyCharm显示缓存可能截断。

## 风险门槛与分析

1000图D1相对D0 AP/APvt下降超过1个百分点，或VT候选覆盖@0.5下降超过2个百分点，暂停正式实验。
VT覆盖按当前评估器verytiny面积区间统计，候选覆盖属于类别无关几何诊断，不是TP/官方召回。
逐图JSONL保留proposal、GT关联、粗细框、无额外阈值预测、预算、密度中心响应、低估项及选中候选组成。
每类评估数组与支持数单独保存。不得将分层过采样子集AP解释为完整test。

三轮结果保留条件：AP≥D0+0.3pp，APvt≥D0+0.3pp，AP75不低于D0−0.2pp，同条件吞吐下降≤5%。
报告八类和vehicle/person VT AP75，不能仅展示低支持类别涨点。
性能工具的候选CUDA事件区间包含CPU发射空隙，不等同于纯kernel时间；两组使用相同记录开销。
D1−D0仅说明组合收益，不区分两个模块贡献。本轮不自动追加24轮或改变默认模型。

## 上游来源和区别

- 论文：https://arxiv.org/html/2505.05741v2
- 官方仓库：https://github.com/RicePasteM/Dome-DETR
- 固定提交：`2dde3bc1946a3e9fad9abd0612b59fc39bd6b861`，本次通过官方API核对。
- 官方LICENSE为Apache-2.0；本次为思想借鉴和独立适配，不拷贝上游训练框架。
- 不迁移MWAS、HGNetv2或D-FINE检测头，不复现官方DRFL公式或原贪心NMS。

固定提交文件SHA-256：

| 文件 | SHA-256 |
|---|---|
| src/zoo/dome/dome_criterion.py | 357e8997241ca12ca146ccaab8d51df28ed563ae5bac53797825ab697b3474bc |
| src/zoo/dome/dome_decoder.py | 34c45f5778d4fed14d006478c89ea7dc5871fffcbc82b6e03f5ca0089410b024 |
| configs/dome/Dome-M-AITOD.yml | 0e5265094b48388b2233efcce2d4cf1fed92d37f3c47e13ed03fe72fdc74b94c |
| LICENSE | c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 |
