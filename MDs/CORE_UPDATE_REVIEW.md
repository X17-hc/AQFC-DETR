# 核心更新验收与第二轮审查

日期：2026-09-07。
实施位置：`D:\PythonProject\AQFC-DETR`。
更新前参考提交：`b47369d24ccba813e6310362297f783f5db5637f`。

## 一、验收摘要

**125项测试通过，1项预期兼容性警告。**

最终测试命令：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
D:/venv/AQFC-DETR/Scripts/python.exe -m pytest tests -q --tb=short -p no:cacheprovider
```

结果：`125 passed, 1 warning in 15.25s`。
警告来自专门构造的“旧checkpoint缺少RNG状态”测试，说明无法保证随机流完全恢复；不是训练报错。

其他检查：

- 五个工具`--help`入口全部返回0。
- 12个实验配置解析和校验通过。
- 五个新增PyCharm XML的解释器、参数和唯一输出目录选项通过检查。
- 更新相关Python文件AST解析通过。
- `git diff --check`通过，仅有Windows LF/CRLF提示，无差异空白错误。
- `pip check`返回`No broken requirements found`。
- 未执行真实数据训练、完整验证、长benchmark、多种子实验。
- 未修改原始DQ-DETR项目、数据、旧日志或权重；没有提交或推送。

## 二、按功能验收

| 功能 | 状态 | 证据与边界 |
|---|---|---|
| 默认AQBA兼容 | 已通过 | 与固定更新前提交比较，参数键、初始化张量和eval关键输出逐项一致 |
| Vectorized密度目标 | 已通过 | CPU/CUDA，chunk1/7/512；空图、重叠、边缘、矩形、有效mask，1e-6容差 |
| Light DW AQBA | 已通过 | 3D/4D输入、密度/预算损失反向、新编码器有限梯度、保存加载 |
| 空间候选 | 已通过核心测试 | 稳定同分、有效索引、无重复、容量再分配、零/NaN密度回退 |
| 模块关闭基线 | 已通过合成整模型 | AQBA/DGFC不在模型参数中，前反向和optimizer step成功 |
| 整模型CUDA链路 | 已通过 | baseline/light_spatial/modules_off_900，128×128合成图，DN损失、反向、optimizer、推理、严格state_dict恢复 |
| 分组数值一致性 | 已通过约定场景 | 组合变体单图900查询，分组/非分组logits与boxes在atol1e-5/rtol1e-4内一致 |
| 两份旧权重 | 已通过加载检查 | best305及pretrain分别对标准/轻量模型warm-start；轻量层显式missing，报告包含独立参数覆盖率 |
| checkpoint策略签名 | 已通过 | 跨变体训练resume被拒绝；标准签名匹配可恢复 |
| 唯一输出目录 | 已通过 | 连续新建不同目录，resume冲突明确报错 |
| 预测导出和类别映射 | 已通过单元测试 | 保留0/7等原ID，空图清单、xyxy→xywh、未知类别报错 |
| 错误分析 | 已通过fixture | 一对一TP、重复FP、无类别GT、密度分桶、独立混淆矩阵 |
| Supervision切片 | 已通过合成fixture | 多切片偏移还原、class-aware去重；实际检测器完整切片评估未运行 |
| Profile | 已通过fixture | 开关关闭不留hook/trace；开启生成trace与摘要，结束移除hook |
| 快速评估后端 | 本机兼容验收受阻 | 官方1.0.2无Windows/Python3.11 wheel；未编译安装，未运行真实双后端数值比对 |
| AP/训练速度收益 | 未验证 | 没有完整训练或真实benchmark，不作涨点/提速声明 |

“整模型测试通过”不等于每种batch/数据分布的所有情况均已穷尽；真实完整一轮仍需要用户手动检查。

## 三、第二轮代码审查：实现与安全

已检查和修正：

1. 关闭AQBA后仍有查询统计字典，不能仅用字典非空判断是否计算辅助损失；现按allocator_enabled和predicted_count分别控制损失与计数日志。
2. 空间模式的同分排序使用稳定token顺序；全局重排前再次按索引整理，避免分组拼接改变tie-break。
3. 密度目标优化仅批量化现有数学定义，reference后端保留，没有静默算法回退。
4. 轻量编码器使用独立键名；保持标准分支不受结构重命名影响。
5. 标准与新变体的训练恢复使用签名，防止表面load成功但策略已变化。
6. 研究配置在模型构建和旧权重加载之前拒绝默认检测warm-start，避免误用旧best305。
7. 可选Supervision不进入模型导入链；测试通过独立子进程确认。
8. 新配置命名空间的未知字段会报错，防止拼写错误被默认值掩盖。
9. 普通训练无profile hooks和新增CUDA同步；profile摘要明确注明嵌套范围不能相加。
10. 源码改动均在AQFC项目；旧默认配置及既有三个PyCharm配置保持原文件。

## 四、第二轮规格审查：明确未交付的效果或限制

### 4.1 快速评估是兼容检查入口，尚非提速交付

适配器目前运行legacy保留原LRP与控制台表格，再执行候选后端比较precision/recall及独立AP25。
阈值为1e-6，缺失值掩码也必须一致，显式请求缺少依赖会报错。
因为当前机器无法安装已发布wheel，不能证明候选后端数值一致，更不能报告加速。
在支持环境还需完成实际fixture/固定预测一致性验证，之后才能考虑真正替代legacy的AP路径；不能把双算当成提速。

### 4.2 可观测性保持诚实

普通预测导出的逐图latency为null，说明未测量，避免异步GPU伪计时。
实际计时在独立runner/benchmark工具中做；真实图像端到端时间不含文件解码，包含预处理、传输、模型与后处理。
数据分层工具是诊断，不替代官方AP/LRP。crowd/ignore仅在诊断统计中排除，不冒充官方crowd匹配。
大规模预测导出和离线分析当前会在内存保存结果；对完整14018图、每图大量候选的导出尚未进行峰值RAM验收。建议先用小规模导出验证，正常训练默认不启用导出。

### 4.3 小尺寸合成测试边界

最初64×64、batch1的测试在标准和新变体都触发原DGFC末层BatchNorm单值限制。
改用128×128合成fixture后通过，没有为极小测试尺寸改动默认DGFC。
用户手动配置的256/288/320尺度不属于该64像素fixture，但完整数据训练的显存/稳定性仍待实际运行。

### 4.4 后续必须手动完成

- 更新版20step短测及完整1个epoch＋完整test评估。
- 所有类别GT/TP支持与LRP输出解释。
- 真实数据峰值显存、模型与整步耗时。
- 基线/变体完整训练及42/43/44重复实验。
- AP+0.3点、查询token−10%和真实延迟的研究门槛。

上述未运行事项没有标记为通过。默认模型不会因新配置存在而升级。

## 五、环境

| 组件 | 验证版本 |
|---|---|
| Python | 3.11.9 |
| torch | 2.7.1+cu118，保持不变 |
| torchvision | 0.22.1+cu118，保持不变 |
| NumPy | 1.26.4，保持不变 |
| OpenCV headless | 4.11.0.86，保持不变 |
| Supervision | 新增0.30.1 |

Supervision依赖先dry-run再安装，新增版本记录于requirements-analysis-lock.txt。
faster-coco-eval-aitod官方1.0.2只列出cp39-manylinux2014_x86_64 wheel与tar.gz；镜像和官方PyPI的binary-only预检均无当前平台匹配项。

## 六、用户入口

先在PyCharm选择 **AQFC-DETR更新版短训练测试**。
完成后自行选择 **AQFC-DETR更新版完整1个epoch及评估**，或先运行基线完整一轮作工程对照。
各次运行生成独立输出目录，不自动复用旧checkpoint。
详细配置、分析工具和研究命令见`CORE_UPDATE_GUIDE.md`。
