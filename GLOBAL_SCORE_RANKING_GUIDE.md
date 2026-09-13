# AI-TOD2 / AI-TOD-V2 跨图分数排序诊断

本入口仅按本项目AI-TOD-V2评估协议验证，不可直接套用于VisDrone。VisDrone必须单独指定其预测、类别和评估协议，不能混用当前8类结果。

## 用途

读取既有预测JSON，用安装的aitodpycocotools官方匹配产生AP75 TP/FP/ignore，再按类别跨图稳定排序。无需torch、checkpoint或GPU。不修改原预测或评估器。

```powershell
D:/venv/AQFC-DETR/Scripts/python.exe tools/analyze_global_score_ranking.py `
  --before outputs/factor_diagnosis_20260909_084529/S1 `
  --after outputs/continuation_20260910_012534/epoch1_subset640 `
  --annotations D:/PythonProject/DQ-DETR/DQ-DETR/data/path/AITODv2/annotations/aitodv2_test.json `
  --image-ids outputs/continuation_20260910_012534/subset_ids.json `
  --diagnostic-errors outputs/paired_head_errors_20260910_v1/prediction_errors.jsonl `
  --output outputs/aitodv2_global_score_ranking_NEW_RUN
```

before/after须为包含log.txt的原实验目录。当前程序依赖上一轮同图分析的错误子类，故必须指向对应预测生成的prediction_errors.jsonl。输出目录必须不存在。

## 固定约束

- IoU=.75，area=all，maxDets=1500，useCats=True；不计算AP25/LRP，不混入普通COCO面积协议。
- 不加置信度过滤/NMS。score分桶仅描述，不寻找最优阈值。
- 同类别按图像ID序列汇集，再稳定score降序，同分行为与原评估器一致。
- 重建101点插值precision须与评估器张量相差≤1e-12；宏AP75与日志差≤1e-6，否则停止解释。
- TP/FP/ignore由官方匹配决定。旧诊断子类只附加在官方FP上，匹配差异另标protocol_matching_difference。本轮未发现此类差异。
- 共同召回使用首次达到指定召回的原始precision，不是插值precision；离散GT导致实际召回可能略高于目标值。
- 同类全局rank不是query ID，不是不同版本同一预测的身份。
- 无GT类别AP为null，不能参与有效类别宏平均。类别算术贡献不是网络模块因果贡献。

## 验证

```powershell
D:/venv/AQFC-DETR/Scripts/python.exe -m pytest tests/test_global_score_ranking.py tests/test_paired_head_errors.py -q -p no:cacheprovider
```

本次29项通过（7项新增跨图测试+22项前一阶段测试），未运行完整训练/整仓库测试。

输出包括两版global_ranks.jsonl、逐类summary、comparison及输入/源码/输出哈希manifest。completed表示完整分析结束；只有核验通过才使用结果报告。
