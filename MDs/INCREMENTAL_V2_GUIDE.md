# AQFC-DETR 增量修复与对照实验使用说明

## Material Passport

- Origin: academic-research-suite / experiment-agent；diagnosing-bugs。
- 版本：incremental_v2，2026-09-13。
- 性质：保留检测权重的工程对照，不是无泄漏SOTA复现。
- 真实训练、完整评估和长benchmark只由用户手动启动。

## 本次改变

正确性修复对所有配置生效：分组Decoder使用实际返回dtype分配散射缓冲区，
并与batch-max路径统一使用显式padding mask，消除已复现的A6000 AMP路径差异；
AI-TODv2仅导出0–7有效类别（9通道参数结构不变）；退化零面积IoU/GIoU不产生NaN；
唯一输出目录增加UUID后缀，防止Windows时钟重复。
这些修复可能改变旧预测，不能拿旧AP直接作为修复后对照。

三种配置位于 `configs/incremental_v2/`：

| 配置 | 工程优化 | 分类监督 |
|---|---|---|
| repaired.py / R | 关闭可选优化 | 原Focal |
| p1.py / P1 | 逐对框损失、合并标量、非阻塞传输 | 原Focal |
| p2.py / P2 | 同P1 | Decoder matching部分质量混合 |

P2不新增推理参数。匹配软目标为detach(IoU²)，正项权重1，负项权重
0.75×detach(sigmoid(logit)²)。与原Focal混合，lambda按成功optimizer更新数升至0.25。
DN、encoder/intermediate分类保持Focal；辅助decoder分类采用质量混合。
参考 [DEIM官方实现](https://github.com/Intellindust-AI-Lab/DEIM)，本实现不是完整DEIM，尚未证明原创性或精度收益。

## 固定起点与时间轴

起点是 `/workspace/AQFC-DETR/outputs/server_update24/20260910_235559_501337/checkpoint0010.pth`。

SHA-256：`898f7542b4c7b96317e857d9df5708ed1de63a8644f9edfaed3f080b2e7e272a`。
已核验epoch=10、完整7009个训练iteration、模型浮点state有限。
启动时再次核验哈希、完整标记、架构字段及100% state形状覆盖；失败就停止，不换权重。

加载普通model，optimizer/scheduler重新建立，EMA关闭。这是warm-start，不是精确续训。
两组各3轮、seed42、batch2、AMP、lr=1e-5、backbone lr=1e-6，学习率恒定。
模型与增强课程按原epoch11–13/总24轮继续，质量预热按新增成功更新数从0开始。
每轮保存；仅末轮完整评估。训练trainval、评估test，禁止按test挑选“最佳”checkpoint。

增量checkpoint包含criterion_progress；恢复时还原质量预热计数。P1与P2不能跨策略resume。
旧原生checkpoint仅在原始课程/Focal兼容路径恢复，并警告修正后的推理语义已生效；
缺少RNG的旧文件仍明确警告不可精确重放。

## PyCharm手动顺序

保持原运行配置，新增6项均使用AQFC-DETR远程解释器、Linux工作目录、GPU环境变量2：

1. **AQFC-DETR历史问题回归与合成验证**：单元和合成CUDA测试，不读取训练图片做训练。
2. **AQFC-DETR修复后20步训练及评估**：真实20步、2个评估batch，只查链路。
3. **AQFC-DETR修复后微调起点完整评估**：共同起点；无需训练。
4. **AQFC-DETR增量优化性能测量**：独立手动启动，不与训练并发测速度。
5. **AQFC-DETR工程提速对照3轮及评估**。
6. **AQFC-DETR质量对齐3轮及评估**。

默认workers=0。benchmark只推荐workers，不会改配置；采用推荐值时，两组必须同时修改。
每个训练入口只运行自身任务，不启动其他组。新输出目录不会覆盖历史实验。
CUDA_VISIBLE_DEVICES只在运行配置中设置，源码不强制物理卡号、不杀其他GPU进程。

## 性能口径

benchmark默认50步预热、200步测量、3次重复；R/w0、P1/w0/2/4/8、P2/推荐workers。
使用同一顺序的trainval图像、验证期确定性800尺度变换，不使用增强。
包含真实临时反向/optimizer更新，但不保存训练权重，不把这些更新算作正式实验。
报告数据等待与H2D至optimizer/标量同步分开计时、images/s、峰值allocated/reserved、AMP跳步。
这是受控step基准，不能替代含Mosaic/Copy-Paste的真实epoch耗时。
`--profile`单独做20步带hook分析，不能与无profile时间混算。失败不重试、不降batch或尺度。
单卡内其他任务竞争必须人工记录，受竞争测量不能作严格速度结论。

## 数据审计

`tools/audit_incremental_data.py --dataset aitodv2|visdrone --data-root ... --output 新文件.json`
只读原图与标注，核查成员ID、类别、框、图像解码/尺寸。零面积GT记录为warning（现有loader过滤），
缺图、坏图、非法引用、类别/划分冲突为blocked。不得静默替换样本。
同一标注文件内检查ID唯一及引用；跨划分用规范化file_name比较成员，不能把各文件独立编号的ID当成全局身份。
本工具不宣称完成像素级重复内容/训练来源泄漏审计。
VisDrone使用现有COCO代理协议，不报告AI-TOD APvt，不冒称官方VisDrone成绩。

## 结果判定模板

| 指标 | 起点R | P1末轮 | P2末轮 | P2−P1 |
|---|---|---|---|---|
| AP / AP75 / APvt / APt | 待运行 | 待运行 | 待运行 | 待计算 |
| 每类GT、TP、AP、LRP支持 | 待运行 | 待运行 | 待运行 | 待分析 |
| 实际训练秒数 / 成功更新 / AMP跳步 | 不适用 | 待运行 | 待运行 | 待计算 |

先确认完整test覆盖、普通模型分支一致、各轮真实更新和checkpoint完整。
再检查P2是否比P1 AP+0.3个百分点、APvt不下降超过0.2点且吞吐损失≤5%。
工程提速15%只是目标，尚未实测。低支持类别波动、无TP导致LRP缺失，不伪造成程序错误。
失败不扩大矩阵、不自动升级默认配置；下一阶段需清洁train→val研究协议。
