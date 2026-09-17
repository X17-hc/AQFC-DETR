# 空间候选等价优化与性能记录修复

## Material Passport

- 技能：diagnosing-bugs；academic-research-suite / experiment-agent。
- 日期：2026-09-15。
- 阶段：实现、回归、无profile短测。
- 状态：实现、相关回归、四组真实无profile短测及离线数值核验已完成。实验描述性分析为ANALYZED，不等于多种子研究验证。
- 范围：仅AQFC-DETR，不修改AQFC-Next、小论文项目、数据集、正式权重或历史结果。

## 实现

### 1. 性能记录

`tools/benchmark_incremental.py`在最后一个测量步骤CUDA同步完成后记录`measured_end_unix`，而非在profiler退出/导出后记录。新增累计测量步骤时间、计时schema版本、是否profile和单独的profile后处理统计。

`util/profiling.py`的摘要schema升级为2：保留CPU/GPU事件类型、inclusive/self时间与解释说明。明确范围统计包含预热，Criterion/Matcher有嵌套，GPU annotation span不等于纯kernel累计计算时间，禁止直接相加。分别记录profiler finalize、trace export、summary aggregation时间。

旧字段的已有数值口径没有偷偷改成不同指标；增加说明字段避免同名事件误合并。不改历史trace或历史结果。

### 2. 候选选择

只替换`models/aqfcdetr/spatial_selection.py`中的逐网格补充候选筛选：

1. 原始joint分数稳定排序保持不变。
2. 已被语义保底选中或非法的token进入零配额的哨兵组。
3. 按网格ID稳定排序，保留每个网格内原有顺序。
4. 根据分组起始位置计算组内名次，一次性选出各组配额以内的token。
5. 后续去重/补齐逻辑、token ID破同分、最终joint排序保持不变。

保留原配额计算和密度平均：避免改变浮点归约顺序导致边界配额变化。不调整loss、查询预算、类别、Encoder/Decoder、AMP生产设置或checkpoint格式。

私有`_quota_additions_reference`冻结原循环，仅用于测试和短测对照；正常训练使用优化实现。不维护两套模型。

### 3. 短测接口

原P1/P2分类损失短测默认接口保持不变。新增`--comparison spatial`，order只接受reference/optimized；两个分支均构建原P2、固定质量lambda=0.25，并从同一普通model权重创建全新模型、optimizer和随机状态。

summary显式标注“P2原候选实现 vs P2优化候选实现”，不把本次收益记成P2相对P1的质量监督收益。

## 验证记录

- 修复前计时placement测试失败：endpoint不在profile上下文中；摘要类型回归失败。
- 修复后本地相关105项通过；服务器相关105项通过。
- 本地全套：270通过、2项明确排除、2条预期fixture警告。
- 排除1：旧测试固定读取`.run/AQFC-DETR_InterleavedGPU.run.xml`，实际运行配置已改中文名。
- 排除2：旧测试固定要求6个`AQFC-DETR_Incremental*.xml`，实际配置已改中文名。
- 当前`test_run_config_catalog.py`单独通过。未为通过旧测试而恢复重复配置或覆盖用户配置。
- 2条警告来自故意不含RNG状态的旧checkpoint fixture，提示不能声称随机流精确恢复。
- CPU/CUDA：随机输入、全同分、全零密度、非有限密度回退、容量不足、矩形网格、比例0/1、空配额等。
- 候选索引、诊断统计、被选memory及其gather梯度逐元素一致。
- 完整合成模型：含DN、主/辅助损失；FP32及AMP前向预测和总检测损失逐元素一致。
- FP32完整梯度沿用atol=1e-5、rtol=1e-4，不放宽；固定测试seed及cuDNN确定性后通过。
- AMP梯度单独审计：同一原实现重复也有原子计算/半精度量化波动；不承诺完整梯度逐bit一致。一次本地审计原实现重复最大差约4.88e-4、优化对照约2.44e-4；这不是泛化误差上界。
- 生产benchmark的cuDNN/TF32设置未因为上述测试而改变。
- 服务器真实CPU profiler小用例通过schema和后处理时间检查；因显式禁用GPU，底层输出设备数不可用提示，不代表GPU训练测试失败。

## 部署与安全

- 8个定向文件部署前均核对旧hash，新增测试同名存在则停止。
- 本地备份：`D:/PythonProject/AQFC-DETR/legacy_artifacts/spatial_perf_backup_20260915`。
- 服务器备份：`/workspace/AQFC-DETR/legacy_artifacts/spatial_perf_backup_20260915T021027`。
- 详细前后SHA-256：同目录`server_deployment.json`。
- 不恢复/清理其他未提交修改，不提交、不推送。

## 本次真实短测

输出：`/workspace/AQFC-DETR/outputs/spatial_equivalent_benchmark/20260915T021247_85d34c20309a`。

物理GPU3，UUID `GPU-7f4ddc8a-8ae2-6547-3300-278e0dd1d384`；batch2、workers2、AMP、800×800、同一500张trainval图像顺序、50步预热+200步测量。

顺序：reference → optimized → optimized → reference。每组重新初始化，无profile、不保存短测更新后的模型。全流程预计12—16分钟，不作为完整epoch实测耗时。

启动前GPU3约12%利用率、463MiB显存；没有发现AQFC训练进程。但容器无法证明宿主机资源完全独占，不因空进程列表宣布独占GPU。

## 方法学检查

按实验分析技能检查11类风险：

| 项目 | 本轮处理 |
|---|---|
| 辛普森悖论 | 同时看AB、BA配对和整体中位数，不只看汇总值。 |
| 生态谬误 | 短测整体吞吐不外推为每张图、每类目标或完整epoch的提升。 |
| 选择偏差 | 固定seed和图像顺序；无增强工作负载不代表完整训练分布。 |
| 碰撞变量偏差 | 不根据本轮速度筛掉慢样本、改变分辨率或预算。 |
| 基础率忽略 | 报告全部测量步骤、成功更新、AMP跳步及查询token。 |
| 均值回归 | 不以历史异常慢速为唯一对照；本轮重新交错测量原实现。 |
| 幸存者偏差 | 保留所有组及异常；未完成/失败不能计作通过。 |
| 多重寻找效应 | 仅试这一种实现；不取多种配置中的最佳值。 |
| 分析自由度 | 启动前固定ABBA、workers2、50+200步，不根据中途结果加跑。 |
| 相关与因果 | 相同配置和设备支持实现对照，但容器资源、时序和CUDA非确定性仍有限制。 |
| 反向因果 | 后端在运行前指定；不因快慢回填后端标签。 |

仅描述性比较，不进行显著性检验；每个后端两次运行不是多随机种子训练，也不能证明AP提升。

## 最终短测结果

2026-09-15 UTC 02:12:47—02:25:37（北京时间10:12:47—10:25:37），全流程约12分50秒。父进程及四个子进程均正常结束，无重试，无正式checkpoint写入。

| 顺序 | 后端（均为P2） | 张/秒 | 平均step毫秒 | AMP跳步 |
|---|---|---:|---:|---:|
| 1 | reference | 2.778547 | 719.801 | 0 |
| 2 | optimized | 2.839221 | 704.419 | 0 |
| 3 | optimized | 2.854574 | 700.630 | 0 |
| 4 | reference | 2.768475 | 722.419 | 0 |

- 原实现吞吐中位数：2.773511张/秒；优化实现：2.846898张/秒。
- 中位数吞吐提升：**2.645983%**；按吞吐倒数换算的时间减少约**2.577776%**。
- 两个相邻配对分别提升：2.183651%、3.109998%。没有只取最佳一组。
- 同后端两次吞吐极差/中位数：原实现0.363159%，优化实现0.539306%。
- 每组200次测量成功更新，加50次预热更新；全队列800次测量更新、1000次总更新，无AMP跳步。
- 四组查询轨迹逐步完全一致：每组有效matching query token总量176,000，batch-max padded matching token总量214,000。不是靠降低K获得速度。
- 四组最大allocated显存相同：14,261.2075MiB（约13.93GiB）；reserved相同：14,594MiB。显存指标不包括CUDA context等全部设备内存。
- 四组配置逐字段一致，普通checkpoint与图像顺序一致，实际输入均为[2,3,800,800]，workers2、AMP，lambda0.25。
- 四组测量窗口分别143.968843、140.892658、140.134846、144.492647秒；与逐step累计仅相差8.72—8.94毫秒。计时口径修复已在实际短测中核验。
- 27—28次有效测量遥测/组，无采样错误、无预定义频率不稳定警告；频率极差约9.13—12.30%。没有据此宣称GPU完全独占。
- 结束后重新核验源码清单、初始化权重、trainval标注SHA-256均未改变；没有在本轮输出目录生成.pth文件。

原始数据保存在`evidence/20260915T021247_85d34c20309a/`；`analysis.json`为数值核验结果，`analyze_benchmark.py`可离线重算。

## 结论与停止点

保留该等价实现：两个配对方向一致，约2.65%的净吞吐增益，索引契约不变，显存未增加。但**没有达到整体15%提速目标**，不表述为解决全部训练效率瓶颈。

收益比profile里某模块CPU范围看起来的占比小，并不矛盾：profile额外放大小算子记录成本；范围里还有GPU等待；这次只优化候选补充筛选，没有消除整个候选模块，也没改可变形注意力。

没有新增完整训练或AP评估，所以不新增精度提升结论，也不把短测速度当完整epoch时间。P2质量监督、查询预算、DGFC、数据协议保持不变。后续若继续提速，应另行批准公共路径优化，不自动扩大矩阵。

## 复测命令

在`/workspace/AQFC-DETR`执行；每次新建唯一输出目录，不保存短测权重：

```bash
CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONIOENCODING=utf-8 \
/opt/conda/envs/AQFC-DETR/bin/python -u tools/benchmark_interleaved.py \
  --data-root /workspace/DQDetr/data/path/AITODv2 \
  --pretrained /workspace/AQFC-DETR/outputs/server_update24/20260910_235559_501337/checkpoint0010.pth \
  --output-dir /workspace/AQFC-DETR/outputs/spatial_equivalent_benchmark \
  --comparison spatial --order reference,optimized,optimized,reference \
  --workers 2 --warmup 50 --steps 200 --amp --telemetry-seconds 5
```

这是原有P2与优化P2的实现对照，不是P1/P2的loss对照。GPU只由环境变量指定，代码未绑定GPU编号。任务已结束，不设置周期任务，不继续运行额外实验。
