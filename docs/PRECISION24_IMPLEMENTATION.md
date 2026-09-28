# F0/F1 24轮联合重构：实现与手动运行

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Version：aqfc_precision_efficiency_24e_v1。
- 状态：工程实现与合成验证；真实训练、吞吐、AP/APvt待验证。
- 目标：同一普通模型完整test AP≥34.0、APvt≥18.5；不是代码交付保证。

## 本轮改动

F0保持原模型结构，F1启用解耦Encoder和局部细化；旧配置默认不启用。
前2个本地epoch全尺度更新6层，epoch2–3更新4层，epoch4起更新2层；后续层仍读取五尺度value，但仅更新P3–P6 query。
P2保留梯度与候选资格，末端新增零输出初始化的轻量上下文残差。
P2/P3原投影特征用于5×5双线性局部采样，分块256个query。
DN query按实际pad_size切除，匹配query与粗框、mask严格对应。
采样坐标停止梯度，粗框输出基准路径与浅层特征保留梯度。
中心偏移与宽高变化有界，末层零初始化。

主Hungarian匹配用粗框；最终回归0.5粗框＋0.5细化框，复用同一匹配。
最终质量监督使用细化框的停止梯度IoU²；辅助层及DN原行为保留。
没有新增几何loss、NMS、检测头或第二次Hungarian。

## 同起点与训练日程

来源：`outputs/scale_v1/s2/20260924_115851_063163_8e60e76c05c3/checkpoint0002.pth`。
SHA-256：`40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928`。
F0旧state全量匹配；F1只允许`local_refiner.`与`transformer.encoder.detail_fusion.`新增键。
任何其他缺失、额外或尺寸错误都拒绝。使用pretrained，不恢复S2 optimizer。

- 24轮、batch2、workers2、AMP、seed42、EMA关闭；trainval→test工程协议。
- 训练640/704/768/800，max1333，S2无预缩小裁剪。
- 原模型phase固定23，Encoder阶段按本地epoch独立推进。
- 原增强和allocator日程未重启，实际Mosaic概率继续由phase23调度，Copy-Paste规则沿用S2。
- 旧非骨干3e-5、骨干3e-6、新参数1e-4；AdamW wd1e-4，clip0.1。
- 500次成功更新预热；跳步不推进；epoch16乘0.1、22乘0.01。
- 每轮保存，只在epoch23完整评估，不按test挑最佳轮。
- 训练1000batch、评估2000batch汇总，磁盘完整控制台与分开计时保留；短冒烟沿用更频繁日志。

## 手动启动顺序

1. `AQFC-DETR联合重构合成验证`。
2. `AQFC-DETR F0原结构20步及评估`。
3. `AQFC-DETR F1联合重构20步及评估`。
4. `AQFC-DETR F1最终Encoder阶段20步及评估`。
5. `AQFC-DETR F0-F1同卡交错性能测量`：ABBA，50预热＋200测量，最终两层阶段，无profile，不保存短测模型。
6. 以上正确性通过后，审查同卡速度和显存，再分别手动运行F0/F1完整24轮。

GPU只在XML环境设置，当前新增入口为GPU2；启动前自行确认资源并统一选择，不代表始终空闲。
工作目录`/workspace/AQFC-DETR`；解释器`/opt/conda/envs/AQFC-DETR/bin/python`。
各入口输出`outputs/precision24`下唯一子目录，不串行自动调度。
若F1最终阶段吞吐提升不足5%，先审查实现与资源状态，不将其称作提速版或直接投入长训练。
基准工具只测最终阶段，不能用其速度代替完整24轮总耗时。

## 恢复

从最近完整epoch checkpoint，用相同F0/F1正式配置、同输出目录，移除pretrained和unique-output-dir，指定resume。
模型结构、阶段、成功更新、参数组名称/成员/峰值LR、scheduler、scaler和RNG保存恢复。
不允许跨F0/F1恢复，冒烟checkpoint不能恢复为正式完整训练。
未实现数据游标，不支持无损epoch中间恢复；不能只手动改start_epoch跳过数据。
评估同样必须用匹配结构配置和resume，使保存的Encoder阶段正确恢复。

## 证据与限制

初始六层、零残差的FP32/AMP恒等测试，以及6/4/2阶段合成DN前反向、参数更新、恢复测试已设置。
新增测试包含粗框匹配与细化质量目标、padding、空GT、奇数/矩形图、分块、严格初始化白名单。
历史运行配置测试按当前真实文件名更新，不修改历史实验与用户选择的GPU。
实际测试数量、环境和部署哈希见交付审查报告；跳过项不能当作通过。

F1−F0只支持组合收益，不拆解成两个模块各自的贡献。test已参与工程设计，达标后仍需清洁协议与单模块消融。
尚未完成真实20步冒烟、无profile测速和两组24轮；不宣称精度或速度已提高。
