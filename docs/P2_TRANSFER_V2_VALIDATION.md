# P2迁移V2：实现与验证报告

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；使用diagnosing-bugs约束复现与验收边界。
- Mode：run（用户已授权的合成和32图参考检查），2026-09-25。
- Implementation：p2_semantic_transfer_v2。
- Verification Status：参考兼容已复测；适配后的学生质量、速度 **NOT_RUN**。
- 结论：可以进入用户手动20步适配检查；不能据本报告直接宣布学生语义退化已消失或开始24轮训练。

## 1. 本次交付

已新增四个独立逐层P2替代器、两阶段独立适配入口、模块权重与恢复契约、32→1000图质量门槛、独立学生ABBA性能入口、五份PyCharm手动配置及使用说明。

原模型默认路径、旧F0/F1和历史数据/权重/输出保留。仅在新版本启用时移除末端补偿、改为固定“两层重型＋四层轻量”；新签名明确记录该结构，不沿用旧6→4→2配方名称。正常学生前向只执行学生链路。

部署改动前核对了服务器原文件SHA；对应原文件在以下目录保留副本，各次修订也有清单：

```text
/workspace/AQFC-DETR/outputs/p2_transfer_deploy/20260925T060548Z/
D:/PythonProject/AQFC-DETR/outputs/p2_transfer_deploy/before_20260925/
```

本地原文件备份目录也包含实施期间新增文件的早期副本；判断原始文件范围以服务器首次部署manifest中的before_sha256为准，不将所有备份文件都误称为任务前文件。

## 2. 自动测试

### 本地完整回归

```text
D:/venv/AQFC-DETR/Scripts/python.exe -m pytest tests -q --disable-warnings
398 passed, 5 warnings
```

最终JUnit：`outputs/p2_transfer_deploy/local_final_tests_v2.xml`。

### 服务器完整回归

结果：**396 passed、1 failed、1 skipped**。没有将其写成“服务器全部通过”。

- 唯一失败：`test_active_remote_launchers_are_safe_and_unique`。服务器原有9组同名旧运行配置（中文文件与英文文件并存），共45个配置但只有36个唯一名称。新增五个P2迁移V2配置自身名称唯一、参数及路径校验通过。遵守保留旧配置的范围约束，没有删除或改名这些历史文件。
- 唯一跳过：`test_supervision_slicer_coordinates_and_merge`；服务器没有可选`supervision`包。没有升级或安装环境依赖，这不影响本次模型/适配路径。
- 最终JUnit：`outputs/p2_transfer_deploy/20260925T060548Z/server_final_tests_v2.xml`。

新P2迁移与现有precision24/stage审计的定向组合：**32项通过**，包含：

- 零残差初值、padding和后续梯度。
- 两分支读取相同更新前memory。
- 教师输入与学生串联两阶段；冻结参数/buffer不变。
- 适配时不运行Decoder；冻结原层保留学生输入梯度。
- 原生CUDA query/value、FP32/AMP、DN、局部细化、检测损失和optimizer step。
- 旧签名不变、新结构固定深度、旧/新跨结构恢复拒绝。
- 不完整适配权重拒绝、正式训练验收门槛、五份XML。
- 子进程AMP开启和关闭均显式传参，不只检查报告标签。

这里只进行了合成训练更新；没有对真实trainval做20步或两轮更新。

## 3. 32图实测：F0 vs 新六层reference

最终有效结果目录：

```text
/workspace/AQFC-DETR/outputs/p2_transfer_v2/validation_fixed_precision/20260925T061133Z_5f636cc2/
```

同32张、583个GT、同S2普通权重、800/max1333、动态查询、spatial、batch1/workers0。两种模式均检查了子进程落盘`config_args_all.json`的实际AMP值。

| 运行精度 | F0 AP | 新reference AP | APvt（两者相同） | proposal覆盖@0.5 | count MAE | 平均K | 导出最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 41.977461 | 41.977461 | 23.752945 | 92.967410% | 4.061769 | 412.5 | 0 |
| AMP | 42.420232 | 42.420232 | 23.775772 | 92.967410% | 4.064278 | 412.5 | 0 |

比较对象是各精度内F0与reference，而不是要求AMP与FP32完全相同。图像/类别顺序、预测数量、分数和原图框导出一致；原始logits/boxes的比较另外由合成测试覆盖。没有将导出一致冒充所有未导出logits逐图一致。

**这些是32图参考兼容成绩，不是完整test成绩，也不是学生成绩。** 不能用这里42.42 AP宣称达到34/18.5目标。

### 本次发现并修复的验证工具问题

第一次自动检查目录：

```text
outputs/p2_transfer_v2/validation/20260925T060608Z_9c1c42b8/
```

该次标为`no-amp`的父任务没有向main显式传递`--no-amp`；main默认AMP开启，实际两组都是AMP。原文件保留，不作为FP32证据。修复后加了子进程参数回归和落盘AMP核验，并在上面的`validation_fixed_precision`新目录重跑，未覆盖旧结果。

另在入口审查中避免了一个配置误判：S2 native_multiscale使用自身固定640/704/768/800变换，而不是继承的legacy data_aug_scales。适配校验检查实际执行的Resize定义，不用不生效字段错误阻止启动。

## 4. 二次审查结论

| 核查 | 结果/边界 |
|---|---|
| 原始文件和历史实验 | 未覆盖数据、权重或旧输出；旧结构默认未升级 |
| 学生结构 | 固定两层重型＋四个独立替代器，没有质量失败时回退参考的分支 |
| 同层更新顺序 | P2替代与低尺度原层使用同一份输入，再拼接 |
| 梯度 | 第二阶段没有对学生memory detach；原层冻结参数仍传输入梯度 |
| 原模型冻结 | 参数与buffer摘要每轮核对；局部细化零残差冻结；仅替代器进入适配optimizer |
| GT用途 | 训练GT仅产生损失空间权重；前向/候选/推理不读取GT |
| 源权重 | 固定S2 SHA；公共参数必须完整对应，新键单列 |
| 恢复 | 仅完整适配epoch，可从第0轮恢复第1轮；不把20步作为正式初始化；不宣称batch级无损恢复 |
| 验收文件 | 32不通过即停止；1000通过且输入哈希复核后才签发；绑定adapter SHA |
| 完整训练入口 | 随机学生或没有通过1000图的模块被拒绝；没有自动启动24轮 |
| 性能 | 仅提供手动ABBA入口，5秒资源采样；无实测速度结论 |
| 测试局限 | 服务器旧配置同名问题未清理，可选Supervision缺失；均明示，不改测试绕过 |

## 5. 下一步与停止点

1. 你手动运行 **AQFC-DETR P2迁移V2适配20步**。总20步分成两阶段各10步，检查有效更新、finite loss和冻结摘要。根据各阶段耗时估算正式适配预算。
2. 成功后手动运行 **AQFC-DETR P2迁移V2独立适配2轮**；从S2重新开始，不继承20步。
3. 在质量验收配置中填入本次`adapter_initialization.pth`；通过32图后才自动继续这次验收任务的1000图，不串接训练。
4. 1000图通过后，再填入`acceptance.json`，手动做独立学生性能检查。

本次到此停止，不训练、不改回六层、不提高K、不放宽门槛。若适配程序正确但质量门槛不通过，应报告迁移未成功，而不是把“无报错”写成真正修复。操作细节与恢复示例见`P2_TRANSFER_V2.md`。
