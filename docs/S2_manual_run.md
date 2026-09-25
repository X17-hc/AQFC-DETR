# S2温和多尺度手动运行

本次仅新增配置及合成测试，不自动启动真实训练。

PyCharm两个入口：

1. **AQFC-DETR S2温和多尺度20步及评估**：20个训练step、2个评估batch。
2. **AQFC-DETR S2温和多尺度3轮及最终评估**：三轮完整训练，每轮保存，只在最后一轮完整评估。

先运行第一个，确认无异常且确实有optimizer更新后再运行第二个。正式训练不恢复冒烟checkpoint。

## 固定条件

- 服务器解释器 `/opt/conda/envs/AQFC-DETR/bin/python`。
- 工作目录 `/workspace/AQFC-DETR`；入口映射到该项目`main.py`。
- GPU环境变量`CUDA_VISIBLE_DEVICES=0`，沿用当前S1模板，可在PyCharm修改，不代表该GPU空闲。
- warm-start为原epoch23：`outputs/p2_legacy24/resume_epoch11_20260916/checkpoint0023.pth`，保留原SHA-256与epoch校验。
- 非resume，不从S1、C0或冒烟终点额外续训。
- batch2、workers2、AMP、seed42、EMA关闭。
- 质量监督0.25、几何辅助项0、主/骨干学习率1e-5/1e-6、phase固定23。
- trainval→test，工程筛查，不作为独立研究验证。
- 独立输出：`outputs/scale_v1/s2_smoke`和`outputs/scale_v1/s2`，自动创建唯一子目录。

## 唯一训练配方变化

`train_transform_mode='native_multiscale'`：水平翻转后，均匀选择短边640/704/768/800，最长边1333，等比例缩放，再归一化。

不执行legacy预缩小或随机裁剪。Mosaic/Copy-Paste及调度继承S1/C0。评估变换不变。
该模式固定上述尺度，不读取`data_aug_scales`覆盖。禁止与`fix_size`或`strong_aug`组合。

模式已进入现有checkpoint策略签名，禁止S1/S2跨模式训练resume。

## 验证说明

本地及服务器AQFC-DETR环境的S1/S2合成回归均32项通过。旧S1 XML测试因写死GPU2、文件枚举包含新增诊断入口而排除；新的S2 XML测试已检查两份入口。
服务器8个部署文件均经过修改前兼容核对及部署后内容哈希校验；备份保存在`outputs/s2_config_deploy_20260924T115517913395Z`。
测试涵盖四尺度、非方形图、空GT、边缘框、标签、原坐标还原、输入不被修改、评估流程不变、配置继承及跨模式恢复保护。
测试中简化checkpoint缺少RNG的警告来自fixture，不代表已完成真实训练续训验收。

真实冒烟与三轮训练均待用户手动验证，不宣称精度或训练速度已提高。C0/C1/C2/S1配置与历史输出保留。
