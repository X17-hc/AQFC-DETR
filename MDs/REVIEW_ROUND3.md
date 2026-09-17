# AQFC-DETR再次审查与运行配置清理

日期：2026-09-13。范围：当前本地/服务器AQFC-DETR；不操作AQFC-Next、小论文数据或训练任务。
本次用户要求为**模型审查＋运行配置清理**，因此以下模型问题为发现和建议，未擅自改动模型算法。
使用diagnosing-bugs的可复现探针方法；测试通过不等于覆盖所有边界，也不能证明AP或速度收益。

## 1. 审查结论

**模型仍存在需要处理的问题，建议优先修复第1项后再进行P1/P2三轮精度对照。**
现有常规合成测试通过，但原单图用例不能覆盖混合查询预算；本次增加独立反例。

### 高优先级：空间候选在训练/推理之间的预算语义不一致

- 位置：`models/aqfcdetr/transformer.py`的非分组候选选择（约630行）和`_decode_grouped_inference`（约372行）。
- 训练：按batch最大K选择，再用query mask保留某图前Ki个。
- 分组推理：直接按该图Ki选择。
- spatial的语义保底数、空间配额随K变化，因此它不是具有前缀一致性的单一排序。
- 固定seed42、64个token、8类、2×2空间网格的实际选择器反例：
  - 直接K=8：`[21,62,4,46,8,25,59,5]`。
  - K=16结果前8项：`[21,62,35,4,46,47,27,44]`。
- 影响当前spatial配置中同batch预算不等的场景；不能简单归为浮点容差问题。
- 建议：训练同样逐图按Ki产生候选，然后仅为张量执行补齐到Kmax；content与reference使用同一索引和mask。
  保留batch-max Decoder/DN，不需要拆成多个训练Decoder。
- 应增加同一encoder输入下混合预算、单图、分组/非分组选择索引的契约测试，覆盖spatial及mixed。
- 尚未测量其实际AP影响；不能宣布这是历史AP下降的唯一根因。

### 中优先级：P2恢复允许不完整质量进度

- 位置：`util/checkpoint.py`约51–55、92行。
- checkpoint含`criterion_progress={}`时，缺少successful_updates仍被默认成0。
- 实际临时checkpoint探针：epoch1成功恢复为start_epoch2，但恢复计数为0。
- 正常由当前main生成的checkpoint会包含计数，因此这不是“所有恢复必失败”。
- 建议：P2严格要求mapping、显式successful_updates和非负整数；在修改模型/optimizer之前验证。

### 中优先级：改变batch续训会让质量预热倒退

- 位置：`util/experiment.py::variant_signature`、`engine.py`质量进度、`util/incremental.py::quality_progress`。
- 签名未记录batch_size，恢复后又以当前DataLoader长度重算预热分母。
- 14,018张图像、已成功更新7,009次：batch2时lambda=0.25；改batch1恢复被接受，lambda变成0.125。
- 建议：持久化预热总更新数并原样恢复，签名校验batch、完整loader长度及相关训练协议；不同策略改用明确warm-start。
- 当前固定batch2、P1/P2各自从共同权重新启动不触发该问题。

### 数值边界：FP16极小框的GIoU不安全

- 位置：`util/box_ops.py::aligned_box_iou/aligned_generalized_box_iou`。
- 实际函数输入完全相同`xyxy=[0,0,0.0001,0.0001]`：FP16返回约0.000327789且梯度非有限，FP32返回1且梯度有限。
- 小面积乘法在半精度下欠流；用半精度tiny钳制分母不能恢复几何精度。原两两实现也存在同类数值风险。
- **尚未证明当前P1/P2正常主链路会输入此种FP16框**；proposal派生参考点通常为FP32，不把此探针当成真实训练已产生NaN的证据。
- 建议将几何面积与比值显式放在FP32计算，并添加低精度输入/梯度回归；不放大容差掩盖问题。

### 未判定为确定性故障

DN空槽作为负类及其注意力行为属于继承的DN策略，是否改变需要独立定义契约，不能擅自删掉这些负样本。
未发现quality_blend匹配类软目标、detach、mask和主/辅助/DN作用范围的新增确定性错误。
未运行真实训练、AP评价、长benchmark；不据此评价精度改善或吞吐目标。

## 2. 运行配置清理

本地原19个入口（18个共享＋1个workspace临时项）精简为9个。服务器还移除了重复的ServerUpdate同名入口。

保留：

1. AQFC-DETR历史问题回归与合成验证。
2. AQFC-DETR修复后20步训练及评估。
3. AQFC-DETR修复后微调起点完整评估。
4. AQFC-DETR增量优化性能测量。
5. AQFC-DETR工程提速对照3轮及评估。
6. AQFC-DETR质量对齐3轮及评估。
7. AQFC-DETR服务器标准版24轮训练。
8. AQFC-DETR服务器更新版24轮训练。
9. AQFC-DETR服务器VisDrone训练（COCO代理验证）。

归档共享配置：BaselineEpoch、LightEpoch、OneEpoch、ServerSmoke、Smoke、SpatialEpoch、Train、UpdateEpoch、UpdateSmoke。
另归档workspace里绑定GPU3、指向历史epoch0的临时评估入口。删除的是IDE列表入口，不是模型训练配方文件。
两份保留的服务器标准/VisDrone入口原GPU环境变量为空，现显式设为2；所有保留配置使用远程AQFC-DETR解释器和Linux工作目录。
本地默认选中“历史问题回归与合成验证”，未启动它。

可恢复目录：两端项目下`legacy_artifacts/run_configs_20260913/`。
内含旧XML及哈希manifest；本地额外保存workspace.before.xml和旧epoch0配置片段。
恢复时仅复制需要的XML回`.run`，不要整份覆盖仍在变化的workspace.xml。
PyCharm若尚显示旧缓存，请关闭后重新打开本项目；未强制结束IDE或训练进程。

## 3. 验证与范围

- 清理前模型回归：本地200 passed。
- 清理后原套件：本地200 passed；服务器199 passed、1 skipped（可选Supervision未安装）。
- 新增运行目录契约测试检查9个入口名称唯一、GPU、解释器、工作目录、入口文件、CLI/配置解析和唯一输出目录。
- 加入目录契约后本地最终201 passed；服务器对应最终结果单独保存于`outputs/review_round3_final.xml`。
- 旧实验12个配置文件及旧运行XML归档仍经过原回归检查，并非删除测试来掩盖错误。
- 模型源码、参数、数据、历史权重和输出均未修改；只有运行配置、相应测试与此报告变更。
- 复现脚本位于本地工作文档目录`probe_aqfc_review_round3.py`，仅使用临时checkpoint和合成张量。
- 模型问题尚待修复；本报告不把审查完成写成“问题全部修复”。
