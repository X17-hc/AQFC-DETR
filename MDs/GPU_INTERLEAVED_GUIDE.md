# GPU竞争验证与P1/P2同卡短对照

## Material Passport

- 技能：academic-research-suite / experiment-agent、diagnosing-bugs。
- 版本：gpu_interleaved_v1，2026-09-15。
- 状态：工具实现与自动测试；真实交错测速尚未执行。已有实验分析为 ANALYZED。
- 不修改模型、质量公式、λ、训练入口、旧配置或正式权重；不自动启动真实训练、评估、测速。

## 先等当前训练和评估彻底结束

当前P2：`outputs/incremental_v2/p2-1/20260914_164640_804919_394f2ad24bfa`。
本次核查时仍在首轮，不能报告最终AP或完整epoch速度。不要与本工具同时运行。
不停止其他任务、不重置GPU、不改容器PID隔离；空进程列表不能证明独占。

## PyCharm手动运行

新增配置：**AQFC-DETR同GPU交错短对照**。

- 解释器 `/opt/conda/envs/AQFC-DETR/bin/python`；工作目录 `/workspace/AQFC-DETR`。
- 运行文件 `tools/benchmark_interleaved.py`，使用普通Run，不使用Python Console。
- 环境 `CUDA_VISIBLE_DEVICES=3`，OMP/MKL线程数2，UTF-8、无缓冲。
- 新目录根 `outputs/gpu_interleaved`，每次再生成唯一子目录。
- 顺序 P1→P2→P2→P1→P2→P1→P1→P2，workers0，batch2，AMP，50预热+200测量。
- 同一seed42抽取500张trainval图像；每组全新Python子进程、模型、优化器与随机状态。
- 使用同一已批准epoch10普通模型，继承现有严格SHA/结构/完整epoch检查。
- 校验实际张量必须为 `[2,3,800,800]`；遇非方图产生其他尺寸时报错，不暗改缩放。
- P2稳定λ=0.25；无增强。短测仅临时更新参数，不保存模型checkpoint。

已有 `tools/benchmark_incremental.py` 原命令仍可用；本次只增加其返回的诊断字段。
没有给训练入口加入GPU编号硬绑定或占用拦截。

## 磁盘输出

| 文件 | 含义 |
|---|---|
| benchmark.json | 完整排期、源码/权重/标注/ID哈希、参数、每组结果、失败现场和汇总 |
| controller.log | 控制台组级摘要、异常与结束状态 |
| image_ids.json | 固定500张图像顺序；前100用于预热 |
| resources_before.json | 开始前的GPU/CPU快照；不宣称独占 |
| 00_p1_w0/console.log 等 | 每组完整stdout与stderr，独立于PyCharm缓存上限 |
| 00_p1_w0/telemetry.jsonl 等 | 默认每5秒记录GPU利用率、功耗、时钟、温度、显存，系统CPU/I/O等待 |
| 00_p1_w0/spec.json、result.json | 子进程初始化来源、测量窗口、逐步shape/K/图像ID/成功更新/时长 |

遥测不可见字段为null并附错误，不补0。CPU为容器可见系统聚合值，不冒称训练进程专属占用。
遥测整个子进程都记录；时钟汇总只使用明确的测量窗口，排除初始化和预热。
端到端组时间单列，不混入同步后的逐step计时；core含H2D、模型、损失、反向和优化器，不称纯模型推理延迟。

## 停止规则与判断

- 子进程失败、非有限损失、连续20次AMP跳步或零有效更新：记录后停止队列，不重试。
- 实际CUDA UUID与nvidia-smi映射不一致：报错；缺失UUID则不宣称设备已验证。
- 每组结束后若遥测不可用、身份未验证、测量窗口SM时钟极差/中位数>15%，暂停后续组。
- 每5秒记录子进程存活和累计耗时；单组超过30分钟标记预算警告，等该组自然退出后暂停队列，不强杀。
- 同一变体已有多次结果的吞吐极差/中位数>10%，暂停后续组。
- 这两个阈值是预先声明的保守有效性警告，不证明有资源竞争，也不删除慢样本；不是GPU占用保护。不会强杀正在执行的组。
- 整体中位数与相邻成对差同时保留。负penalty表示P2更快，不截成0。
- P2下降≤5%且资源记录稳定：暂不改loss。下降>5%且稳定：另行手动profile。
- 不自动追加workers2、不重跑完整P1、不自动微调或改默认模型。

工具故意不在缺失遥测时自动继续凑齐八组。先检查记录，再由用户决定新的手动启动时机。

## 条件满足后才使用的两种命令改动

1. workers0稳定且P2代价≤5%：在新运行中改为 `--order p1,p2 --workers 2`，其他条件相同。
2. 稳定代价>5%：在新运行中用 `--order p1,p2 --workers 0 --profile`。profile固定1预热+20测量，输出独立trace，只定位瓶颈，不作为速度结论。

二者均沿用唯一目录。不改变当前训练配置；不把workers收益写成模型创新。

## 完整P2完成后的离线核验

运行 `tools/summarize_gpu_repeat.py`，传入 `--r0 --p1 --old-p2 --new-p2 --output-dir` 五个路径参数。
它仅读JSON，输出新Markdown/JSON，核对3×7009步、有效更新、阶段11—13、三个checkpoint存在及14018评估计数。
**summary_complete只代表汇总字段齐全**，不是进程已退出、checkpoint内部正确或完整ID一致的证明。
最终还须检查进程退出、checkpoint元数据、八类PR/面积数组与ID集合。没有预测文件时不伪造逐GT分解。
新旧P2 AP或APvt差>0.3个百分点只是排查信号；同seed重复不是多随机种子实验。

## 仍待完成

- GPU3完整三轮与最终评估结果核验。
- 用户手动启动真实交错短测；因此尚不能判断质量loss真实代价是否≤5%。
- 如需续训，先处理已知空criterion_progress及warmup分母随DataLoader改变的恢复风险；本次不修改正在运行的训练代码。
- 不承诺涨点、接近SOTA或VisDrone泛化；保留旧慢速记录并注明资源混杂。
