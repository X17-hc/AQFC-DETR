# 最终epoch23模型：几何辅助项与同起点三轮对照

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Mode：经用户授权实现、合成验证；真实训练手动运行。
- Version：geometry_v1，2026-09-17。
- Evidence：实现与定向测试已通过；三轮训练、完整评估及精度收益未验证。

## 1. 为什么只加这个变量

既有1000图离线复核显示，very-tiny邻近框的相对中心误差、宽高偏大现象在32图以外仍存在；这是设计假设的依据，不是唯一根因证明。
不同尺寸档并非都偏大，因此不统一缩框，不修改推理、候选预算、类别分数或后处理。
保留当前quality-blend、L1/GIoU/NWD及AQBA/DGFC；只补充最后一层Decoder的匹配框监督。
本机制是待检验的工程适配，不预先宣称原创或涨点。

## 2. 定义与梯度保护

预测和GT都是归一化cxcywh。用每张增强后图像的 `target['size']=[H,W]` 转成像素，不能使用原图尺寸或batch padding尺寸。

```text
dx = max(GT_w_pixels, 4)
dy = max(GT_h_pixels, 4)
r = [(pred_cx-GT_cx)/dx, (pred_cy-GT_cy)/dy,
     (pred_w-GT_w)/dx,   (pred_h-GT_h)/dy]
L_geometry = sum_matched(mean_4(SmoothL1(r, beta=0.1))) / num_boxes
L_total = L_original + eta * L_geometry
```

`num_boxes`沿用原Criterion的GT数量归一化（分布式时为各rank平均GT数量，下限1），不再额外除一次匹配数量。

- FP32计算，GT与尺度分母停止梯度；prediction保留梯度。
- Smooth-L1对残差的一阶导数绝对值不超过1；4像素下限避免极小框分母趋零。
- 归一化预测坐标的导数仍包含W/H，不能声称任意网络参数梯度都小于1。继续沿用原全局梯度裁剪0.1，并观察grad_norm、AMP跳步。
- 不硬截断残差，不用“越界就零梯度”的方式掩盖误差。
- 只使用最后一层既有Hungarian匹配索引，不重复匹配；无效query由原Matcher排除。
- 不给DN、encoder中间输出、其他Decoder辅助层增加此项。
- 空匹配返回可微零；默认权重0时完全不执行新损失。

eta最大0.05。新优化器成功更新数为u，完整训练loader长度为N：`eta=.05*min(1,u/N)`。
第一步eta=0；成功完成N次更新后达到0.05。AMP跳步不推进u，因此有跳步时预热可能略超过首轮。
保存成功更新数和固定N；续训缺少进度或N变化时报错，而非偷偷重新预热。

## 3. 两组固定条件

| 项目 | C0原损失 | C1几何项 |
|---|---|---|
| 初始化 | 同一epoch23普通model | 同左 |
| 新损失最大权重 | 0 | 0.05 |
| 其他解析后配置 | 完全相同 | 完全相同 |
| 分类监督 | quality_blend，lambda固定0.25 | 同左 |
| 训练 | trainval，3完整epoch，batch2，AMP，workers2，seed42 | 同左 |
| 优化器 | 新AdamW；主LR1e-5、骨干1e-6；三轮恒定 | 同左 |
| 推理及评估 | 动态spatial；800/max1333；最终完整test；无TTA/新增NMS | 同左 |
| 保存 | 每轮checkpoint，EMA关闭，不按test选best | 同左 |

来源：`/workspace/AQFC-DETR/outputs/p2_legacy24/resume_epoch11_20260916/checkpoint0023.pth`

SHA-256：`c447367e2d19411319a990985e0127af8db08bfb7eec0329b2065b095ccc64eb`

服务器已核验epoch=23、epoch_complete=True、train_iterations=7009、非smoke。两组均strict warm-start，不加载旧optimizer，不是精确续训。
模型结构与参数键不变；服务器两组均严格加载成功，参数数目49,896,920。
不能把C0终点用作C1起点，也不能把C1与原epoch23直接比较后把继续训练收益算成辅助项收益。

新局部epoch为0/1/2，原模型课程固定在23/24末期：teacher=0、allocator weight=0.75、Mosaic继续原末期概率，Copy-Paste保持原配置。此处是**重复原末期状态**，不是宣称续接epoch24—26。
分类质量预热显式关闭（warmup_epochs=0），两组从第一步保持lambda=0.25。
空间、查询、DN、增强及其他损失配置均来自当前完成24轮的配方。

## 4. PyCharm手动启动

打开本地 `D:/PythonProject/AQFC-DETR`，选择：

1. `AQFC-DETR C0原损失对照3轮及最终评估`
2. `AQFC-DETR C1尺度归一化几何3轮及最终评估`

建议串行运行，不在同一GPU同时启动两组。两组都绑定远程AQFC-DETR解释器、Linux工作目录 `/workspace/AQFC-DETR`，关闭Python Console运行。
GPU只在运行配置环境变量 `CUDA_VISIBLE_DEVICES=3` 设置；需要更换时两组一起修改，不改源码。不保证该卡独占。

配置文件：`configs/geometry_v1/c0_3e.py`、`configs/geometry_v1/c1_3e.py`。
新输出根目录：`outputs/geometry_v1/c0/`、`outputs/geometry_v1/c1/`，每次自动生成唯一子目录。
每组生成checkpoint0000/0001/0002；只在局部epoch2评估。
没有自动启动、周期任务或提交推送。

正式启动前可复制相应运行配置做手动20步工程测试：把max-train-steps改20、max-eval-steps改2，并通过 `--options epochs=1 val_epoch=[0]` 设置只跑一轮。该测试仍使用完整loader长度作为预热分母，所以不能用它判断辅助项最大权重下的精度；对应合成测试已在0.05下执行。smoke输出不能作为正式续训起点。

## 5. 断点恢复

发生中断时只能从该组已完整完成的epoch checkpoint恢复；不支持恢复半轮已消耗样本。
复制对应运行配置：删除 `--pretrained ...` 和 `--unique-output-dir`，添加 `--resume 该组checkpoint路径`，output-dir指定该组原唯一目录，其余配置不变。
保留epochs=3，入口按checkpoint epoch+1接着跑剩余轮数，不是额外再跑三轮。
C0与C1训练签名不同，禁止跨组resume。C1必须恢复geometry_warmup_steps和successful_updates。

## 6. 验证清单与限制

- 新增19项测试全部在本地通过，包括两组真实模型的128像素合成前后向/optimizer step；无真实图像。
- 定向回归40项本地通过；服务器CPU36项通过、4项CUDA主动跳过。
- 服务器两组最终权重严格加载通过；旧epoch23签名在新增默认字段后保持兼容。
- 本地全套：319通过、3失败。失败分别是旧配置文件数量写死、旧交错测速英文文件名已改名、旧总配置测试把CPU离线配置CUDA_VISIBLE_DEVICES=-1误判。不是新增几何损失失败；未为追求全绿恢复过时配置或把CPU配置改为GPU。
- 服务器全套CPU：287通过、34跳过、1失败。失败为7组历史运行配置重名；排除新C0/C1后仍存在。未擅自删除服务器历史配置。
- 本次新增配置单独验证：XML合法、Linux工作目录、同一checkpoint、参数无冲突、仅geometry_loss_weight不同。
- 一次本地受限权限全套pytest长时间无输出，已只结束本次测试进程；随后正常权限全套结果如上。该无输出尝试不记为通过。
- 真实20步、两组完整三轮和完整test尚未运行；本次没有占用服务器GPU训练。

服务器部署前原文件备份：`legacy_artifacts/geometry_v1_preimages_20260917T124906.tar`。
本地原文件备份：`C:/Users/hechang/Documents/ChatGPT/论文/AQFC_geometry_v1_preimages`。

## 7. 跑完后如何判断

唯一主要对照是C1−C0：AP提高至少0.3个百分点，APvt下降不超过0.2个百分点，同时检查AP75、person/vehicle及全部类别支持数、重复框与定位误差。
记录每轮7009步、optimizer成功更新与AMP跳步、geometry_weight、loss_geometry及未加权值、grad_norm、纯训练与评估耗时。
服务器竞争影响需单列；新增辅助项不承诺提速，期望开销小于5%仍需同条件测量。
单seed、trainval→test属于工程诊断，不能作为显著性、泛化或接近SOTA的独立证据。

按空闲时历史1.5小时/epoch估计，两组三轮纯训练约9小时，加两次完整评估约3—5小时；资源竞争时明显延长。训练后将两个输出目录交给后续分析，不自动扩大实验。
