# 第1项修复：候选按实际查询预算选择

日期：2026-09-13。只修复再次审查报告第1项及具有同一根因的mixed候选路径。
使用diagnosing-bugs流程：先在实际Transformer训练调用处重现失败，再修复并回归。

## 根因与修改

旧训练/非分组路径使用Kmax选候选后截前Ki项，spatial/mixed的配额又依赖K，因此不能保证等于直接按Ki选择。
修复后传入已经由合法proposal数裁剪的query_counts；预算不同的图像分别按Ki选择，再补齐索引到Kmax。
补齐部分重复第一个合法索引，但只属于padding，仍由原query mask排除注意力键、匹配、分类损失和后处理。
content、encoder辅助输出及4D参考框使用同一索引。单次batch-max训练Decoder及DN逻辑不变。
所有图预算相同时继续原批量选择路径；分组推理本就使用Ki，不改其选择算法。
spatial诊断按原batch顺序记录实际选择数量，不再把Kmax的空间配额误记为小预算图的配额。

不改变查询档位、教师、检测头、损失、权重形状或可训练参数；未修复第2–4项。
没有新增默认关闭的开关掩盖此正确性修复。

## 可复现证据

- 修复前：`test_real_training_uses_each_images_budget[spatial]`在真实CUDA Transformer、DN开启、两图预算8/16条件下失败。
  断言在Decoder前直接比较索引，排除了Decoder浮点差异；固定seed42可复现。
- 修复后测试覆盖semantic/fused/mixed/spatial、同分、矩形空间网格、非有限proposal、padding及索引gather/梯度。
- 真实预算300/500/900/1500各自与单图选择完全一致；合法proposal不足的裁剪预算和零密度回退也覆盖。
- 合成CUDA训练验证混合预算、DN损失、主/辅助损失、有限反向、optimizer step和后处理。
- 同一encoder batch下，分组/非分组的入选初始框严格相同；FP32最终预测使用atol=1e-5、rtol=1e-4。
  不要求不同backbone batch组成之间完全等价，也不借此声称AMP所有设备逐位相同。
- 本轮不运行真实数据训练、完整评估或benchmark，不声称修复后AP必然提高。

最终自动回归：本地213 passed（17.56秒）；服务器212 passed、1 skipped（23.24秒）。
跳过项为服务器未安装可选Supervision的切片测试，与本修复训练路径无关。
两端报告均保存为`outputs/per_image_budget_fix_pytest.xml`；两个缺RNG警告来自明确构造的旧checkpoint测试。
不同预算需要逐图候选调用，实际速度影响尚未benchmark；这是正确性修复，不作为已测提速收益。

## 权重和实验兼容

模型state_dict不变，批准的epoch10普通model仍可作为严格完整模型warm-start。
correctness_revision变为`incremental_v2_per_image_selection`：旧incremental_v2训练checkpoint不能直接伪装成相同策略resume。
需要继续旧实验时应明确新建warm-start实验；本次P1/P2手动配置仍从批准的共同epoch10权重开始，无需更换起点。
新版本自身保存恢复按现有签名规则继续使用。
原历史无correctness_revision的普通Focal恢复兼容警告路径未扩展，历史兼容限制仍以checkpoint加载器为准。

## 交付和后续

保留原9个PyCharm入口。先手动运行“修复后20步训练及评估”，通过后再开始P1/P2公平对照。
不要把本修复引起的训练语义收益计为质量混合损失的独立算法收益：两组必须使用相同修复版本。
部署前保存源码快照与逐文件原/新哈希；没有删除数据、权重或历史输出，也没有Git提交/推送。
