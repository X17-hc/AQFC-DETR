# AQFC-DETR：pretrain24 冻结实验基线

本分支归档 **2026-10-15 启动、从 `pretrain_model.pth` 初始化的24轮实验**。训练实现保持实验冻结目录原样；不包含后续门控修复、效率改动或 `improvement_v1`。

| 项目 | 原始记录 |
|---|---|
| AP | **31.9618** |
| APvt | **16.5551** |
| AP50 / AP75 | 68.3627 / 25.7250 |
| 轮次 | `epoch=23`，零基编号，即第24轮 |
| 数据 | AI-TOD-V2，trainval训练，test评估，14,018张测试图像 |
| 实现 | ResNet-50，5尺度，6层encoder/6层decoder，`legacy_joint_v2` |
| 训练 | batch 2，seed 42，AMP，24轮，workers 2，模型EMA关闭 |
| 最终评估 | `eval_backend=legacy`；常规模型分支；无TTA |
| 开始时间依据 | 运行清单 `2026-10-15T04:28:53.588777+00:00`，北京时间12:28:53 |
| 末条指标时间 | `2026-10-16 14:51:54.152651`，原字段未附时区，不据此补造结束时区 |

检测指标统一乘100展示；完整原始浮点值见 [metrics.jsonl](evidence/pretrain24_20261015/run_pretrain24/metrics.jsonl) 的第24条记录、`test_bbox_metrics`。指标数组含 **AP25**，不能套用标准COCO数组下标。

这是一份历史源码与证据发布，**没有重新训练或重新推理**，不承诺不同环境下逐位数值复现。本实验最终test结果不能当成独立验证集上的最佳模型筛选结果，也不代表后续修复已获验证。

## 版本来源与权重身份

- 冻结源码：`/workspace/AQFC-DETR/experiments/aqfc_pretrain24_20261015/code`。
- 原始输出：`/workspace/AQFC-DETR/experiments/aqfc_pretrain24_20261015/run_pretrain24`。
- 初始化：`/workspace/AQFC-DETR/weights/legacy/pretrain_model.pth`，SHA-256：
  `e6c3723b0f82c7aa6a6954576ad4e572bab1f87b57281069dd1814afedc16abe`。
- 第24轮对应输出文件：`checkpoint0023.pth`；另有滚动保存的 `checkpoint.pth`。只核实了文件存在与文件属性，本次未读取大型检查点，也未计算它们的哈希或检验载荷，因此不声称两文件逐字节一致。
- `pretrain_model.pth` 是**初始化输入**，不是这次24轮训练产生的权重。其更早训练来源在原清单中为 `unknown`。
- 24轮正式训练从该初始化重新开始，未从20步工程短测检查点续训。

原始源码清单180项均通过采集校验；本分支纳入175项，排除1个编译动态库和4个build目录下的Python构建副本。编译源文件完整保留。见 [发布清单](evidence/pretrain24_20261015/published_source_manifest.json) 和 [原始清单](evidence/pretrain24_20261015/source_manifest.json)。

其他配置文件是冻结目录内原有配置，不能据其存在认定本次运行启用了这些设置。**本次唯一正式配置为 `configs/mature_pretrain24.py`。**

## 环境与依赖

依赖安装文件和 `cocoapi-aitod` 编译源来自父提交 `058abd66d17480d2e1a30f68bc807d40e0b39b0b`，不是原180项冻结清单的一部分。包内原许可证继续保留。4个评估器Python文件已与服务器当前安装文件按LF换行归一化核对一致，见 [核对结果](evidence/pretrain24_20261015/evaluator_source_verification.json)。

采集时服务器环境：Python环境路径 `/opt/conda/envs/AQFC-DETR`；torch 2.4.0+cu124，torchvision 0.19.0+cu124，numpy 1.26.4，scipy 1.16.1，Pillow 11.0.0，timm 1.0.19，aitodpycocotools 12.0.3，pycocotools 2.0.8。**这是当前环境观测，不是已经证明的训练时完整环境锁。**

在独立的兼容Linux/CUDA环境安装相应PyTorch、项目依赖并编译算子：

```bash
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt
python -m pip install --no-build-isolation -r requirements.txt
cd models/aqfcdetr/ops
python setup.py build install
cd ../../..
```

这些是复现准备命令，本次发布没有执行安装或GPU测试。现有依赖清单部分版本未固定，不能称为完整环境锁；CUDA Toolkit、驱动和编译器须与所选PyTorch环境匹配。

原控制脚本设置过包含 `faster_coco_eval_aitod` 的外部 `PYTHONPATH`。本次正式配置使用legacy评估器；静态检查显示快速评估器仅在显式选择 `faster_aitod` 后加载，因此此分支的基线命令不要求该可选包，不改用快速评估口径。

## 训练与评估入口

先准备原始初始化权重与AI-TOD-V2数据，检查权重SHA-256。数据标注哈希见 [原始运行清单](evidence/pretrain24_20261015/run_pretrain24/experiment_manifest.json)。权重和数据不随Git发布。

在仓库根目录执行以下命令；路径替换为实际绝对路径，`CUDA_VISIBLE_DEVICES` 由运行者选择可用GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
python -u main.py \
  --data-root /absolute/path/to/AITODv2 \
  --pretrained /absolute/path/to/pretrain_model.pth \
  --device cuda --amp --seed 42 --num_workers 2 \
  --max-consecutive-skipped-steps 20 \
  --config configs/mature_pretrain24.py \
  --output-dir /absolute/path/to/new_pretrain24_run \
  --options joint_subset_ids="$PWD/evidence/pretrain24_20261015/subset_ids.json"
```

原配置的中间子集评估安排在epoch 0/3/7/12，最终全量评估在epoch 23。上述命令仅将原服务器绝对路径指向归档的同一份子集ID文件，不修改模型或训练配方；该文件采集时的哈希和来源已保存，原运行清单未记录其当时哈希，因此历史逐字节身份仍有此证据边界。

现有最终检查点的独立评估入口如下；它是将来由使用者运行的命令，本次未执行：

```bash
python -u main.py \
  --data-root /absolute/path/to/AITODv2 \
  --resume /absolute/path/to/checkpoint0023.pth \
  --eval --device cuda --amp --seed 42 --num_workers 2 \
  --config configs/mature_pretrain24.py \
  --output-dir /absolute/path/to/new_pretrain24_eval
```

不要无参数启动 `main.py`：冻结实现的默认路径与默认初始化未为发布重新改写。原控制脚本保存于证据目录，仅用于追溯；含原机器路径且有启动/写入行为，不是可直接运行的便携启动器。

## 版本差异与已知边界

发布前核对结果：当前服务器294个源码文件，本地309个；换行归一化后284个相同、9个内容不同、服务器独有1个、本地独有16个。详见 [三方版本说明](docs/VERSION_COMPARISON.md)。

特别是服务器当前 `util/legacy_joint.py` 缺少冻结版本针对本次初始化哈希及22个额外缺失参数的兼容逻辑。此次保留冻结实现，不把当前主目录或后续修复冒充本实验版本。权重加载覆盖及缺失项详见 [迁移记录](evidence/pretrain24_20261015/run_pretrain24/checkpoint_migration_report.json)。

冻结版本中既有门控、padding处理等行为均未修复。未来改进必须另建分支并独立验证。本分支只对应一次运行，不提供统计显著性或多随机种子稳定性结论。

## 发布验证

运行 `python tools/verify_baseline_release.py` 可仅用Python标准库核对175项源码哈希、Python语法、配置继承路径、24轮记录及最终具名指标，不启动模型。

证据目录保留原始配置、运行清单、迁移记录、状态与指标；说明和清单由本次发布整理生成。原始实验没有Git提交号，本次Git提交是**事后归档标识**，不能写作训练当时提交号。其他历史实验与工具可在父提交及main查阅，不作为本分支当前运行入口。
