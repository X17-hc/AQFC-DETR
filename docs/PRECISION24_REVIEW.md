# AQFC-DETR F0/F1完整24轮重构交付审查

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Version：aqfc_precision_efficiency_24e_v1，2026-09-25。
- 验证范围：源码实现、合成前反向、初始化、恢复、启动配置、文件部署。
- **尚未验证：真实数据20步、无profile速度、两组24轮及最终AP/APvt。**
- 目标AP≥34.0/APvt≥18.5仍为研究目标，不是已达成绩。

## 1. 已部署

本地`D:/PythonProject/AQFC-DETR`与服务器`/workspace/AQFC-DETR`共36个必要文件。
部署前旧文件哈希全部匹配，未覆盖服务器独有源码。原DQDetr、AQFC-Next、数据、历史模型与输出不变。

备份：

- 本地：`D:/PythonProject/AQFC-DETR/outputs/precision24_deploy/1790309048296001400`。
- 服务器：`/workspace/AQFC-DETR/outputs/precision24_deploy/1790309073495081753`。

每个目录保存部署manifest及修改前文件。源码未自动提交或推送。
本报告目录的`manifest.json`列出逐文件变更哈希；SHA规范化CRLF为LF，部署内容采用LF。

## 2. 模型与训练机制

已实现显式query/value分离、6→4→2渐进Encoder、P2零初始化融合，以及原P2/P3局部5×5采样联合细化。
P2不detach、不删除候选，DN切分及分组后的matching query与框保持对应。
使用真实输入尺寸换算像素坐标，连续采样align_corners=False，采样坐标停止梯度。

匹配使用粗框，最终回归粗/细各0.5，最终quality_blend使用细化框IoU²，辅助层/DN原规则保留。
新模块初始化隔离随机流；新增参数独立学习率且不重复进入optimizer。
原phase23与本地epoch解耦，500次成功更新预热，AMP跳步不推进。
恢复保存/核对结构阶段、参数组成员、峰值LR、RNG与更新数。拒绝跨F0/F1恢复；EMA未纳入本轮，配置明确拒绝启用。

## 3. S2真实权重审计

服务器权重SHA：`40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928`。
实际epoch=2，完整epoch=true，smoke=false，7009训练迭代，trainval/test。

| 项目 | F0 | F1 |
|---|---:|---:|
| 全部参数数量（含冻结参数） | 49,896,920 | 51,085,404 |
| 新增可训练参数 | 0 | 1,188,484 |
| 允许缺失的新state键 | 0 | 34 |
| 共享state与S2逐元素相同 | 是 | 是 |

这是CPU模型初始化核验，没有读取真实图像或执行训练。原普通模型未换成EMA模型。
明细见`precision24_s2_start_audit.json`。

## 4. 测试结果及真实限制

| 环境/范围 | 结果 |
|---|---|
| 本地实际项目，PyTorch 2.7.1+cu118，完整回归 | **378 passed**，5条warning |
| 服务器，PyTorch 2.4.0+cu124，新功能测试 | **12 passed** |
| 服务器实际项目，完整回归 | **376 passed、1 skipped、1 failed** |

服务器唯一失败为历史运行配置名称重复：39个配置文件对应30个显示名称，保留了9个旧英文文件名副本。
这不是本轮模型数值失败。新增7个入口自身无重名且解析测试通过。未擅自删除、重命名或覆盖服务器旧配置。
因此不能写“服务器全量测试全部通过”。本地PyCharm入口无此重名问题。
服务器唯一skip：可选Supervision未安装，切片工具fixture未运行；本轮训练和推理不依赖该工具，未升级环境。

首次隔离回归中，6项旧测试在未修改的原项目也失败（旧文件名、固定GPU号、混入诊断入口的文件计数、命名解释器SDK_HOME序列化）。
已修正测试发现规则，保留配置契约与用户原运行设置。最终本地实际项目连同旧权重迁移测试全部通过。

### 数值与梯度

- 初始六层、融合/细化零残差：FP32和AMP，F1相对关闭两个模块的原结构，logits/boxes最大绝对差均为0。
- 6/4/2层各完成合成前向、主/辅助/DN/allocator损失、反向和参数更新；包含混合空GT、奇数非方形图。
- 500/300混合预算：Decoder组输出与细化器query输入逐元素相等，验证散射次序，不依赖数值容差。
- 服务器FP32分组与补齐路径：logit最大误差9.54e-7，粗框与细化框约5.96e-8，满足atol1e-5/rtol1e-4。
- AMP跨批形状比较：logit最大差0.00390625，粗框约6.11e-7、细化框约1.16e-6。**该比较不满足FP32级logit容差，不伪称严格数值等价。**
- AMP的索引对应检查仍是严格相等；未扩大FP32容差。半精度不同batch/sequence形状的路径误差单独保留。本计划正式评估batch=1。
- 测试包含零初始化第一步、后续新增层梯度、坐标stop-gradient、输出粗框梯度、有效query mask、严格warm-start及跨结构恢复拒绝。

机器可读证据：`local_final_regression.xml`、`precision24_server_final.xml`、`precision24_server_deployed_full.xml`，以及同名服务器log。

## 5. PyCharm手动顺序

新增7个入口（比计划多提供F0真实冒烟，满足对照前置检查）：

1. AQFC-DETR联合重构合成验证。
2. AQFC-DETR F0原结构20步及评估。
3. AQFC-DETR F1联合重构20步及评估。
4. AQFC-DETR F1最终Encoder阶段20步及评估。
5. AQFC-DETR F0-F1同卡交错性能测量。
6. AQFC-DETR F0原结构24轮及最终评估。
7. AQFC-DETR F1联合重构24轮及最终评估。

环境GPU2仅为当前入口设置；启动前确认空闲并统一GPU，不代表永久独占。
工作目录Linux绝对路径，使用现有服务器AQFC-DETR环境。
所有训练入口固定S2源权重，唯一输出目录；正式训练不要从冒烟checkpoint续训。
先完成2–4，再运行5。异常、零更新或最终阶段吞吐改善不足5%，暂停投入长训练并检查原因。
ABBA不保存更新后的模型，日志、图像顺序、GPU UUID及遥测落盘，不把资源竞争收益当模型收益。

## 6. 二次审查与停止点

- 旧默认模型不启用新模块；原forward接口和检测协议保留。
- 新模块不读取GT；GT仅在既有训练损失中使用。
- 新增query切片依赖实际DN长度，不硬编码100。
- 主匹配不因细化重新执行Hungarian；粗/细监督共享索引。
- 训练/结构时间轴分开，成功更新预热不受AMP跳步推进。
- 真实训练、最终评估、速度复测没有自动启动；没有停止其他人的GPU任务。
- 输出、数据、checkpoint未清理或覆盖。重复运行配置和可选依赖限制明确保留。

当前可进入**用户手动真实冒烟与测速阶段**，不是已完成两组完整训练或已证明精度提升。
F1−F0仅解释组合净收益；即便达到34/18.5，仍先属于test工程结果。
