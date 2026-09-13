# 同图同GT离线检测误差分析

此工具只依赖NumPy；绘制案例时延迟导入Pillow。不导入模型、torch或Supervision，不改变训练及官方评估入口。

## 运行

```powershell
D:/venv/AQFC-DETR/Scripts/python.exe tools/analyze_paired_head_errors.py `
  --before outputs/factor_diagnosis_20260909_084529/S1 `
  --after outputs/continuation_20260910_012534/epoch1_subset640 `
  --annotations D:/PythonProject/DQ-DETR/DQ-DETR/data/path/AITODv2/annotations/aitodv2_test.json `
  --image-ids outputs/continuation_20260910_012534/subset_ids.json `
  --output outputs/paired_head_errors_NEW_RUN `
  --image-root D:/PythonProject/DQ-DETR/DQ-DETR/data/path/AITODv2/images/test/images `
  --expected-images 1000 --expected-gt 90536
```

`--before/--after`可指向唯一预测导出目录或其上一级实验目录。输出目录必须不存在。省略`--image-root`时不生成案例。没有自动重试、目录清理或默认模型参数修改。

## 口径

- 用原始像素xywh计算IoU，不裁剪框。非法类别、非有限数据、非正宽高、ID不一致、query mask或导出数量不一致均报错。此协议要求导出条数等于实际K，不接受经过置信度筛选的导出。
- 当前协议遇到ignore/crowd直接报错，不静默剔除，也不声称兼容COCO crowd匹配。
- GT按annotation ID排序；预测按score降序，同分保留该图原导出顺序。最大IoU同分时匹配最小GT ID。
- `prediction_index`是图内原导出序号，不是模型query索引。`rank`为score排序后从1开始的位置。
- 逐GT几何覆盖允许多对一；四种匹配模式严格一对一。两种统计不能混称召回。
- 阈值0.50/0.75及score≥0.25均包含等号；邻近定义IoU≥0.10。
- `geometry`针对最佳同类IoU框。中心偏差除以GT宽高，width/height ratio是预测尺寸/GT尺寸，aspect ratio relative error是预测宽高比/GT宽高比−1。没有同类预测时为null；同类框存在但全无重叠时仍记录其最佳值，汇总解释应排除这些任意远框。
- `low_score`与`ranking_inconsistent`可以重叠，另存交集。局部高分邻近框可能正确检测另一个GT，不能直接认定为分类头错误。
- 无score过滤的FN也可同时标记低分与匹配竞争；标记是描述，不是互斥FN分解。score025模式可用几何不足/低分/竞争核对FN。
- FP互斥分类顺序：重复、类别混淆、定位不足、类别和定位同时不足、背景。FP类别按预测类别；尺寸按该错误关联GT分桶，背景为unassigned，不按预测框面积伪装GT尺寸。
- `groups`中的GT分层统计与`prediction_groups`中的预测侧统计分开，不能任意拼接各分组TP和FN。
- 前300条仅截断已导出的预测，不等价于固定300查询，也不是重新计算AP。

## 文件

- `paired_gt.jsonl`：image_id+annotation_id配对的两版特征及四模式匹配结果。
- `prediction_errors.jsonl`：每版每条导出预测及四模式错误标签。
- `summary.json`：GT、类别、尺寸、密度、epoch0预算档的计数及迁移矩阵；逐图预测统计。
- `cases.json`、`cases/*.png`：定位下降、局部排序标记、改善各最多4张；先按变化排序，再按image/GT ID，同图不重复。案例有目的选取，不是随机代表样本。
- `manifest.json`：命令、阈值、环境、输入和源码哈希、状态；只有completed可交付为完整结果。

## 测试

```powershell
D:/venv/AQFC-DETR/Scripts/python.exe -m pytest tests/test_paired_head_errors.py -q -p no:cacheprovider
```

本次22项新增测试通过；连同原错误分析回归测试共23项通过。未运行整仓库测试或训练。
