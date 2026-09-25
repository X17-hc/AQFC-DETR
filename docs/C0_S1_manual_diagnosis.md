# C0—S1同图同GT定向诊断

PyCharm选择 **AQFC-DETR C0与S1同图同GT定向诊断** 后手动运行。
环境默认GPU2，启动前自行调整`CUDA_VISIBLE_DEVICES`；代码不绑定GPU，不检查或终止其他任务。

固定C0、S1各自三轮最终普通模型，SHA256保护；800/max1333、batch1、AMP、workers0、seed42。
只评估，使用`--resume --eval --no-pretrained`严格加载模型；不是恢复训练。
先核验原1000张清单，再补齐包含四类稀有VT GT的图像。按GT而非预测挑选，VT面积包含64边界。
每个模型仅推理一次合并清单，不评估全test，不训练，不改query/loss，不覆盖旧输出。

执行顺序：CPU预检 → C0评估及全部proposal导出 → S1评估及全部proposal导出 → CPU配对分析 → 分层汇总。
原工具每次只保存前32张NPZ，但本入口另保存每张图的`proposals.jsonl`，不会受该32张上限限制。
候选观察器仅在前向完成后读取输出和GT，不返回替代输出。原预测仍受模型后处理Top-K截断；不能推断未导出的框。

输出根：`/workspace/AQFC-DETR/outputs/scale_v1/paired_diagnosis/<唯一时间戳>`。

运行完成后提供这个完整目录，重点文件：
- `pair_manifest.json`：完成/失败状态、来源哈希、GPU清单、epoch0=C0与epoch1=S1映射。
- `sampling.json`、`image_ids.json`：基础与补充成员，不混同于完整test。
- `C0_console.log`、`S1_console.log`、`paired_console.log`：完整子阶段控制台记录。
- `C0/proposals.jsonl`、`S1/proposals.jsonl`：有效入选encoder框和逐GT候选IoU。
- `paired/summary.json`、`paired/paired_gt.jsonl`、`paired/prediction_errors.jsonl`：几何/排序/一对一错误。
- `targeted_summary.json`、`targeted_gt.jsonl`：原1000张、额外图像、官方VT类别与全部稀有VT单独汇总，关联两版proposal最佳IoU。

错误会停止后续队列，不自动重试、不降尺度；每次重新启动创建新目录。无需再给C0/S1新增训练运行配置。
可在参数末尾临时添加`--check-only`仅做CPU预检；正式诊断必须移除此参数。
结果属于富集子集工程诊断，不是完整test AP或严格速度对照。
