# AQFC-DETR 运行与训练修复验证报告

验证日期：2026-09-06。此报告替代 2026-08-29 的初始环境判断；实地检查确认本机有 CUDA 11.8、RTX 4060 和完整 AI-TOD-V2 路径，并已完成以下实际运行。

## 结论

在新建的 `D:\venv\AQFC-DETR` 独立环境中，项目已完成 CUDA 算子编译、完整模型前向/反向、真实数据短训练、密集样本压力测试、checkpoint 续训及独立评估。应使用 `local8gb` 配置在本机训练；原论文大分辨率配置没有得到全密度训练显存验证。

## 本次修复

1. CUDA 扩展适配当前 Tensor API：`scalar_type()`、`data_ptr()`、`is_cuda()`。
2. AMP 下可变形注意力的 value、sampling locations、attention weights 使用一致 dtype。
3. Decoder 自注意力实际传入 target padding mask，禁止有效查询关注 padding 查询。
4. EMA 边界用于路由，辅助损失改用原始预测边界，恢复 boundary head 梯度。
5. DN 正样本匹配不再遗漏每图最后一个 GT。
6. 不同 encoder 有效长度的 batch 正确填充 proposal 索引，避免形状不一致。
7. DataLoader 支持 workers=0；训练 worker 每 epoch 重建，避免 persistent worker 持有过期增强概率。
8. allocator warmup 按真实 loader 长度计算，不再硬编码 iteration 总数。
9. 新增短训练/评估步数上限；去除隐式覆盖用户 resume/pretrained 的行为。
10. AMP scaler 跨 epoch 复用并保存/恢复；优化器跳步时不更新 EMA/逐步 LR；记录真实更新步数。
11. 修复命令行单元素列表解析；保存 best checkpoint 和未评估 epoch 的训练日志。
12. 从源码修复并编译 AI-TOD mask API，明确声明 NumPy 的 C 级所有权标志。
13. 修正效率工具：使用真正 decoder 执行 token，而不是将 padded 模式误记为每图预算之和。
14. 增加 Windows 新环境安装、CUDA 编译和低显存训练脚本。

## 实际通过的验证

| 项目 | 实际结果 |
|---|---|
| 环境独立性 | venv 基于 Python 3.11.9 新建，不继承其他 site-packages；CUDA 可用 |
| 依赖一致性 | `pip check`：No broken requirements found |
| 单元/回归测试 | 27 passed，0 skipped |
| CUDA float 前向 | 与 PyTorch 参考实现一致；最大绝对误差 4.66e-10 |
| CUDA double 前向 | 最大绝对误差 8.67e-19 |
| CUDA 数值梯度检查 | channels=4，三个可微输入全部通过 |
| 完整模型训练 | 合成 batch=2，主损失、DN、allocator 损失前向/反向和 optimizer step 通过 |
| 分模块梯度 | allocator、boundary_head、density_head、DGFC、decoder 均有限且非零 |
| 同档推理一致性 | 300 查询 grouped 与 padded 输出通过 atol=1e-5 / rtol=1e-4 |
| 混合档推理一致性 | batch 内 300/500/900/1500，所有有效位置输出与 padded 对照一致 |
| 旧最佳权重迁移 | 按参数量覆盖率 99.8817%，无形状不匹配，新增参数明确列于报告 |
| 真实数据训练 | 20 iteration，11 次真实 optimizer 更新，9 次 AMP 初期安全跳步 |
| checkpoint 续训 | 从 epoch 0 恢复到 epoch 1，再训练 3 次，optimizer 步数 11→14，scaler 128→128 |
| 小规模评估 | 训练内验证和独立 `--eval` 均通过，每次限制 2 张图 |
| 密集样本训练 | image_id=9736，原始 1729 标注、有效 1727，320×320、1500 查询、DN，更新通过 |
| 密集样本显存 | 峰值 allocated 6637 MiB，约 6.48 GiB |

27 个 pytest 并不等同于 27 个完整模型集成测试。真实 CUDA 和数据验证由单独工具、训练入口执行，证据分开保存。

## 短效率冒烟（不能当作论文基准）

同一 RTX 4060 Laptop、新环境、随机合成输入 256×256、batch=2、FP32、5 次预热 / 20 次计时，CUDA synchronize 计时：

| 执行方式 | 平均延迟 ms/batch | decoder tokens/batch |
|---|---:|---:|
| 固定 900 | 195.79 | 1800 |
| batch-max 动态 | 162.72 | 600 |
| 分组动态 | 173.11 | 600 |

本次随机输入两图均分到 300，因此分组相对 batch-max 的 token 降低为 0%，且分组略慢；不能声称分组在此测试中加速。四档混合一致性测试的理论实际 decoder tokens 为 3200，padded 对照为 6000，但完整性能结论必须在真实数据、足够预热与样本量下重新测量。

## 文件与日志

- `outputs/runtime_smoke_final/`：20 iteration 日志、checkpoint、最佳 checkpoint、迁移报告。
- `outputs/runtime_resume_verified/`：实际续训结果，optimizer 步数达到 14。
- `outputs/runtime_eval_verified/`：独立评估结果。
- `outputs/runtime_verification/`：环境摘要、测试输出、CUDA 数值验证、密集样本报告和短效率结果。
- `TRAINING_WINDOWS.md`：启动、续训、低显存配置和重装步骤。

输出目录、二进制扩展和权重不纳入源码提交。原模型代码和数据未被本次修复改写。

## 验证边界与剩余风险

- 没有执行完整 24 epoch、多随机种子或全量 AP 评估；不保证 AP 提升或长期收敛。
- 低显存配置改变输入分辨率/增强/EMA，不能等价替代原论文配置的精度实验。
- 8GB 压力测试覆盖单张最密集图，不是所有输入长宽、增强与显存占用组合的上界证明。
- 真实短训练用 workers=0、单 GPU；多进程 worker、DDP、多 GPU 和 VisDrone 全流程尚未实测。
- 部分 torch/torchvision 旧接口仍有弃用警告，不影响本次固定环境；不要未经验证升级 PyTorch/NumPy。
- 短效率结果只验证工具运行与计量路径；未运行正式 50 预热/200 计时的全矩阵。
