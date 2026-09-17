# AQFC-DETR 增量修复与二次审查报告

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；diagnosing-bugs。
- 版本：incremental_v2，2026-09-13。
- 执行范围：现有AQFC-DETR；不涉及AQFC-Next和小论文源码。
- 状态：工程验证记录；真实20步、完整微调、完整评估及benchmark均未自动执行。

## 1. 发现、复现与修复

| 编号 | 复现/证据 | 处理 | 语义边界 |
|---|---|---|---|
| F1 | 服务器原基线175通过、1失败、1跳过；light_spatial AMP分组差最大0.0078125 | 分组与batch-max使用相同显式padding mask；探针统一mask后原用例通过 | 不放宽容差；旧分组预测可能改变，必须重建起点评估 |
| F2 | 最小CPU用例：encoder Half、decoder Float，散射报dtype不匹配 | 每层缓冲区以实际decoder输出dtype创建 | 消除降精度/类型冲突，不宣称这是F1唯一原因 |
| F3 | 实际AI-TOD模型工厂返回的PostProcess.valid_category_ids为None | 工厂仅允许输出0–7，保留9通道参数 | 旧第9通道可占据Top-K；没有重跑历史AP，不能定量归因 |
| F4 | 退化零面积GIoU产生NaN，失败测试已复现 | 仅保护零分母，两两与逐对路径一致 | 正常框数学定义不变 |
| F5 | Windows连续unique_output触发FileExistsError；冻结时钟回归 | 时间戳追加UUID后缀 | 保留目录防覆盖及resume冲突保护 |
| F6 | 新3轮warm-start若沿用epoch0会重启教师/增强课程 | 独立阶段偏移11、总阶段24；成功更新计数独立保存 | 明确是新微调，不恢复跨策略optimizer |

本次实现中发现的dict形式expected_args恢复兼容问题也已修正；原回归重新通过。
二次审查还纠正了新增审计工具的跨划分身份错误：COCO ID在各标注文件中独立编号，
因此跨划分改用file_name；新增独立重编号测试。非有限/结构错误框仍阻断，不能按零面积warning处理。
原生checkpoint策略签名新增loss/阶段/工程开关和correctness_revision，旧兼容路径明确警告。

## 2. 历史问题状态

| 历史问题 | 当前处理/验证 |
|---|---|
| AMP密度饱和、count exp、边界EMA污染 | 既有数值修复保留；饱和半精度、NaN/Inf及梯度回归 |
| 教师/预测路由单位、DN关闭教师丢失 | 既有修复保留；计数单位、教师退火和DN测试 |
| raw边界头失去梯度 | 既有直接raw-head监督回归；并未重新初始化AQBA |
| density mask与奇数尺寸 | 既有真实encoder mask、reference/vectorized等价回归 |
| padding query进入损失/输出 | 实际Criterion/PostProcess零梯度和输出测试；分组新增F1/F2 |
| Mosaic取整坐标、Copy-Paste遮挡、空图递归 | 既有回归保留，不改原图/GT |
| 坏图被替换为另一张 | 既有明确异常机制保留；新增原image ID数据审计 |
| EMA计数buffer被平均、普通/EMA混淆 | 既有状态与显式评估分支测试；P1/P2仅普通model |
| 短测伪续训、遗漏epoch/RNG/scaler | 既有严格恢复回归；新增质量计数保存/跨变体拒绝 |
| AP25、LRP、尺寸/类别协议 | 保留原官方表格和独立AP25；VisDrone仍标COCO代理 |
| 输出目录冲突 | F5修复；不删除旧checkpoint、不自动resume |
| 推理token下降却未加速 | 不是确定性bug；提供手动实测，不报告未测提速 |

无TP导致LRP不可用、单轮低AP、batch组成影响等不被伪装成已修复代码故障。
不宣称所有输入组合、分布式训练或完整24轮均已验证。

## 3. 实现与实验隔离

- 逐对GIoU/NWD在空匹配、单框、多框上验证值和梯度，NWD常数仍0.03。
- 标量传输按设备合并；原日志指标保留；不删除非有限loss检查。
- NestedTensor支持pin_memory和non_blocking，原调用接口仍可用。
- P2只改decoder主/辅助matching分类；DN与encoder/intermediate分类不改。
- 质量目标/权重detach，padding梯度为0，lambda=0与原FP32损失一致。
- 原Focal稳定计算明确转FP32，属于数值稳定路径，不把AMP旧值差异声称为位级等价。
- 两组从同一epoch10普通model完整加载，lr1e-5/backbone1e-6、3轮、EMA关闭。
- 配置、signature、manifest和checkpoint共同携带阶段、损失和恢复计数。
- 6份PyCharm配置经过XML与真实parser/config冲突检查；无自动启动链。

## 4. 验收证据

- 本地实际项目：200 passed，2个预期旧checkpoint缺RNG警告；无跳过，15.36秒。
- 证据：`outputs/incremental_v2_final3_pytest.xml`。
- 本地环境pip check通过；未安装/升级依赖。
- 服务器最终实际项目：199 passed、1 skipped、2 warnings，20.80秒；`pytest_final3.xml`。
  跳过项为`test_supervision_slicer_coordinates_and_merge`：服务器未安装可选Supervision，
  因此不声称服务器切片工具已验收；核心训练、合成CUDA前反向和更新测试通过。
- 修订AI-TOD审计passed（100条警告包含trainval重复），VisDrone审计passed（3条警告）。
- 原指定checkpoint：epoch10、7009 iteration、epoch_complete=true、模型浮点state有限。
- 权重SHA-256：`898f7542b4c7b96317e857d9df5708ed1de63a8644f9edfaed3f080b2e7e272a`。
- 服务器实际构建P2、严格加载指定普通model成功：49,896,920个参数，参数覆盖率100%，输出有效类别0–7。
- 本地和服务器pip check均通过，未升级依赖。

### 数据审计证据

服务器证据目录：`outputs/incremental_v2_validation_20260913_01/`。

| 数据划分 | 图像条目 | 无正面积框警告 |
|---|---:|---:|
| AI-TODv2 train | 11,214 | 40 |
| AI-TODv2 val | 2,804 | 9 |
| AI-TODv2 trainval | 14,018 | 49 |
| AI-TODv2 test | 14,018 | 2 |
| VisDrone train | 6,471 | 3 |
| VisDrone val | 548 | 0 |

四个AI-TOD划分共42,054条图像记录已解码（含trainval重复成员，不是42,054张独立图片）；
VisDrone共7,019张已解码。未发现缺图、无法解码或图像尺寸不符。
AI-TOD按file_name确认train∩val为空、trainval等于二者并集、trainval∩test为空。
不据此宣称像素级同内容无泄漏。原审计的ID重叠误报保留在`aitod_audit.json`，不能用它证明数据泄漏。
修订审计另存`aitod_audit_corrected.json`，引用原完整解码证据并核验标注SHA-256未变。
零面积警告不修改原始GT；现有训练loader过滤无正面积框，评估仍使用原标注。

## 5. 备份、部署与回退

首次部署前本地源码快照：`outputs/incremental_v2_deploy_1789292207228654400/source_before.tar.gz`。
服务器首次快照：`outputs/incremental_v2_deploy_1789292244827554353/source_before.tar.gz`。
各部署目录包含逐文件before副本与原/新SHA-256清单；后续修补也单独备份。
部署只接受目标文件仍匹配已审查原哈希，服务器独有文件不在包内，不覆盖。
没有删除数据、历史日志或权重，没有Git提交/推送。
回退前先停止用户自行运行的相关任务并备份新输出，再按manifest恢复具体源码；不要删除整个项目。

## 6. 未运行与下一步

真实短测、起点完整test、P1/P2各3轮及末轮完整test、速度benchmark待用户手动启动。
当前不宣称AP提高0.3、吞吐提高15%或接近SOTA。
执行顺序及指标模板见 `INCREMENTAL_V2_GUIDE.md`。
