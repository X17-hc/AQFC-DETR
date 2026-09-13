# AQFC-DETR 服务器部署与验证记录

日期：2026-09-10。服务器：219.216.64.62:32880，账号 root（不记录密码）。

## 1. 当前结论

环境、源码、数据加载、权重加载和运行配置的静态验证已完成；**尚未完成 GPU 前反向及真实数据短训练，不能宣称完整训练已通过**。

四张 A6000 均有占用，尚待用户确认可用 GPU 编号。本次未启动 GPU 训练、完整评估或后台训练队列，未结束其他进程。新增 PyCharm 配置暂将 CUDA_VISIBLE_DEVICES 留空，确认分配后必须填入获准使用的物理卡号，才能启动 CUDA 训练。

## 2. 目录与独立环境

| 用途 | 路径 |
|---|---|
| AQFC 项目 | /workspace/AQFC-DETR |
| 独立解释器 | /opt/conda/envs/AQFC-DETR/bin/python |
| 历史小论文模型 | /workspace/DQDetr |
| AI-TOD2 数据 | /workspace/DQDetr/data/path/AITODv2 |
| VisDrone 数据 | /workspace/DQDetr/data/path/VisDrone |
| 本次检查证据 | /workspace/AQFC-DETR/outputs/server_deployment_20260910 |

未克隆、复用或修改 DQDetr 虚拟环境。独立安装得到：

| 组件 | 已验证版本 |
|---|---|
| Python | 3.11.13 |
| PyTorch | 2.4.0+cu124 |
| torchvision | 0.19.0+cu124 |
| NumPy | 1.26.4 |
| Pillow | 11.0.0 |
| CUDA 编译器 | 12.4.99 |
| 独立 GCC/G++ | 11.2.0 |
| aitodpycocotools | 12.0.3 |
| pycocotools | 2.0.8 |
| MultiScaleDeformableAttention | 1.0，Linux CPython 3.11 本机编译 |

panopticapi 已安装。pip check 通过；完整依赖版本存于证据目录 requirements-server-actual.txt。Supervision 未安装，不影响主训练。

原系统 CUDA 11.0、G++ 7.5 不适合直接作为本次工具链。已安装独立 CUDA 12.4 与 GCC 11.2；编译所需 cuSPARSE 等头文件取自 AQFC 环境自己的 NVIDIA pip 包。脚本 scripts/build_ops_server.sh 固定该方式，默认编译 A6000 的 sm_86。编译时隐藏全部 GPU，不代表 GPU 算子数值测试已通过。

## 3. AI-TOD2 图片修复与数据清单

缺失文件：images/trainval/images/03615.png。

已从同数据集 images/train/images/03615.png 补入。复制前核对图像尺寸及该图片的框、类别、面积、ignore/crowd 标注内容；源图完整解码成功。仅在目标不存在时创建，复制后验证 SHA-256 一致：

c02ab7c9a1333d67f45086aeeafa32e7c3093ad4988a46ea3512cc3ef2415350

未修改标注。记录：data_repair.json。

| 数据集 | 划分 | 图像数 | 标注数 | 按标注路径检查的缺图数 |
|---|---|---:|---:|---:|
| AI-TOD2 | train | 11,214 | 301,534 | 0 |
| AI-TOD2 | val | 2,804 | 75,091 | 0 |
| AI-TOD2 | trainval | 14,018 | 376,625 | 0 |
| AI-TOD2 | test | 14,018 | 376,121 | 0 |
| VisDrone | train | 6,471 | 352,018 | 0 |
| VisDrone | val | 548 | 40,137 | 0 |

这是全量路径存在性检查，不是所有图片的完整解码扫描。AI-TOD2 trainval/test、VisDrone train/val 均另行通过真实 Dataset 取样、变换及有限值检查，记录见 real_dataset_samples.json。全量清单和标注哈希见 data_inventory.json。

## 4. VisDrone 加载与验证修复

已复现旧实现错误：dataset_file=visdrone 时 PATHS 未赋值，导致 UnboundLocalError。

本次修改：

1. 统一 VisDrone 路径解析：VisDrone2019-DET-train/images、VisDrone2019-DET-val/images，对应 annotations_coco 下原转换标注。
2. 明确只支持 train/val；不把 trainval/test 偷换为 val。
3. 保留原类别 ID 1–10。训练排除 ignore/crowd；非忽略目标出现未知类别时明确报错。
4. 保留已有 12 通道检测头。后处理仅在通道 1–10 排序并输出原始类别 ID，防止无类别意义的 0/11 通道泄漏到结果。
5. 运行清单记录 VisDrone 标注，不再误找 AI-TOD 标注；变体签名加入该数据契约，避免不兼容续训。
6. 分离 VisDrone COCO bbox 代理评估器，使用标准 COCO 面积分桶、maxDets=100，不再套用 AI-TOD 的 APvt/APt/LRP 名称。

**重要限制：此代理评估不是官方 VisDrone DET 评估协议。** ignore 在评估用 GT 副本中以 crowd 方式处理，但不能据此声称实现了官方全套忽略区域规则。不可直接将代理 AP 与小论文 VisDrone AP 作严格数值比较。正式复现论文须另行接入并核对官方评估器。原标注未被改写。

## 5. 小论文参数参考与本次运行方式

核对来源：/workspace/DQDetr/logs/DQDETR_ver7/0414-mosaic+fullv2/config_args_all.json。

新增服务器标准版参考其完整训练条件：

| 项目 | 服务器 24 轮配置 |
|---|---|
| batch | 2（单卡） |
| epochs | 24 |
| 训练短边尺度 | 480、512、544、576、608、640、672、704、736、768、800 |
| max_size | 1333 |
| 主学习率 / Backbone 学习率 | 1e-4 / 1e-5 |
| weight decay | 1e-4 |
| 学习率衰减节点 | 13、23 |
| EMA | 开启，0.9997 |
| DN number | 100 |
| AMP | 开启；初始 scale=32，属于运行稳定性设置 |
| checkpoint / 评估 | 每轮保存；第 24 轮统一评估 |
| 数据协议 | trainval → test，engineering_check |

Mosaic、Copy-Paste 保留 AQFC 24 轮配置的调度，而不是本机 8GB 的关闭增强配置。workers=0 优先确保服务器首次链路稳定，尚未做吞吐调优。

注意：这些是**参考历史参数的 AQFC 实验**，不是小论文精确复现。AQFC 标准版为 standard/reference/fused/Tanh；更新版为 light_dw/vectorized/spatial/Tanh。两者使用旧检测权重 warm-start，并非从本机累计 6 轮 checkpoint 继续训练，也没有恢复历史 optimizer。旧权重来源的完整训练数据仍须单独审计，不能当作无泄漏的研究初始化。

历史 18 轮 finetune 配置另用 1e-5/1e-6、最大短边 900、EMA 0.9999，不能与上面的 24 轮参数混为一套。

## 6. 权重与源码补齐

已将服务器历史目录内的两个权重复制到 AQFC weights/legacy，未覆盖源文件。大小与 SHA-256 已逐一校验，记录 server_copy_manifest_20260910.json。

- dqdetr_best305.pth：236,322,319 字节。
- pretrain_model.pth：226,833,878 字节。

使用 dqdetr_best305.pth 构建并 warm-start 的实际结果：

| 变体 | 模型参数量 | 含 buffer 的 state-dict 元素加载覆盖率 |
|---|---:|---:|
| 标准 AI-TOD2 | 58,994,648 | 99.8812% |
| 更新 AI-TOD2 | 49,896,920 | 98.9453% |
| 标准 VisDrone | 58,996,190 | 本次只构建，不加载 AI-TOD 检测权重 |

此表覆盖率来自 coverage_by_numel，不冒充“仅参数覆盖率”；逐键缺失、形状不匹配和参数专用统计以迁移 JSON 为准。高加载比例不代表未加载轻量分支已训练好。

源码分批部署前备份每个被替换文件。备份位于 deployment_backups/20260910_141553、20260910_142608、20260910_143319。未清理原输出、数据或历史 checkpoint；未执行 Git 提交或推送。此次未将本机全部历史输出和累计 6 轮 checkpoint 重新上传。

## 7. PyCharm 配置

本机 D:/PythonProject/AQFC-DETR/.run 新增四项，保留旧配置：

| 显示名称 | 配置文件 | 用途 |
|---|---|---|
| AQFC-DETR服务器更新版短训练测试 | configs/server/aitodv2_update_smoke.py | 20 个训练 step + 2 个评估 batch |
| AQFC-DETR服务器标准版24轮训练 | configs/server/aitodv2_standard_24e.py | 标准对照 |
| AQFC-DETR服务器更新版24轮训练 | configs/server/aitodv2_update_24e.py | 轻量＋空间候选 |
| AQFC-DETR服务器VisDrone训练（COCO代理验证） | configs/server/visdrone_standard_24e.py | train→val，无旧检测权重 |

SSH SDK：SSH (sftp://root@219.216.64.62:32880/opt/conda/envs/AQFC-DETR/bin/python)。

工作目录：/workspace/AQFC-DETR。SDK 的目标项目根目录已从错误的 /root/AQFC-DETR 修为 /workspace/AQFC-DETR；原全局 SDK 文件已在本机文档工作区备份。

四项配置都通过 XML、实际 argparse 参数解析、配置继承校验、数据与初始化文件存在性检查。使用 -u、UTF-8、唯一输出目录；不附带自动训练任务。

**目前尚不能直接点击运行：CUDA_VISIBLE_DEVICES 暂为空，等待 GPU 分配。** 确认卡号后填写该字段；模型使用 cuda，即可映射到获准的物理 GPU。不要因当前空闲量较多就抢用他人 GPU。

本次是磁盘配置检查，未在 PyCharm 界面点击运行。运行中的 IDE 可能尚未重载外部 XML 修改；如列表未刷新，应重新打开项目并核对 SSH SDK、映射与运行参数。不要用“当前文件”无参数模式运行服务器训练，因为 main.py 的默认参数仍为本机 8GB 用途。

## 8. 验证结果与证据边界

| 检查 | 结果 |
|---|---|
| pip check、核心包导入 | 通过 |
| 原生算子 Linux 编译及模型导入 | 通过 |
| 新增 VisDrone 5 项测试 | 5 passed |
| 全量 tests 最终回归 | 170 passed，7 skipped，1 warning，10.93 秒 |
| 标准/更新/VisDrone 三模型构建 | 通过 |
| 两版 AI-TOD2 warm-start | 通过 |
| 真实 Dataset 四划分取样 | 通过 |
| 四项服务器 XML/CLI/路径 | 通过 |
| GPU 算子数值与前反向 | 待确认 GPU 后运行 |
| AI-TOD2 20 step + 2 batch | 尚未运行 |
| 完整 epoch / 24 epoch | 尚未运行 |
| 官方 VisDrone 指标一致性 | 未实现、未验证 |

7 个跳过：3 个 CUDA 特性测试、3 个 CUDA 集成测试、1 个可选 Supervision 测试。唯一 warning 来自故意构造的缺 RNG 历史 checkpoint 测试，不是本次真实训练告警。

初轮两项失败已修复并重跑：JetBrains 等价路径的测试适配；固定 Git 历史源码测试改用 SHA-256 锁定的原始夹具，未扩大数值容差。

证据文件：full_tests_v2.log、dataset_tests_v2.log、build_ops_v2.log、model_build.json、各变体 migration.json、data_inventory.json、real_dataset_samples.json、requirements-server-actual.txt。

## 9. 下一步唯一必要输入与验收

请先确认可使用的 GPU 编号（0、1、2、3）。之后：

1. 更新四项运行配置的 CUDA_VISIBLE_DEVICES。
2. 运行 CUDA 特性/集成测试。
3. 使用相同服务器更新版参数执行 20 step + 2 batch，不自动扩展为完整训练。
4. 核对 optimizer 实际更新、AMP 跳步、有限损失、显存、保存与评估结果。
5. 通过后再由用户启动 24 轮任务。

直到上述 GPU 与短训练完成前，“服务器已具备构建和加载条件”不等于“已验证正常训练”。

