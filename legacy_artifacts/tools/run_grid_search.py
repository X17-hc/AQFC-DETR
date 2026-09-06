import os
import subprocess
import itertools

# 1. 定义论文中 \lambda_1, \lambda_2, \lambda_3 的搜索空间
# 论文默认值: lambda1=1.0, lambda2=0.5, lambda3=0.5
# 实验所用：coverage_loss_coef 0.1 interval_loss_coef 0.05 spacing_loss_coef 0.2
# 建议在默认值上下浮动进行网格搜索
# lambda1_vals = [0.5, 1.0, 2.0]  # Coverage Loss (\lambda_1)
# lambda2_vals = [0.1, 0.5, 1.0]  # Spacing Loss  (\lambda_2)
# lambda3_vals = [0.1, 0.5, 1.0]  # Count Loss    (\lambda_3)
lambda1_vals = [1.0]  # Coverage Loss (\lambda_1)固定
lambda2_vals = [0.5]  # Spacing Loss  (\lambda_2)固定
lambda3_vals = [0.1, 0.5, 1.0]  # Count Loss    (\lambda_3)

# 2. 基础的训练启动命令（请根据你原本正常的训练命令进行替换）
# 例如你通常运行: python main_aitod.py --config_file config/DQ_5scale.py --batch_size 2 ...
base_cmd = [
    "python", "main_aitod.py",
    "--config_file", "config/DQ_5scale.py",  # 你的配置文件路径
    "--batch_size", "2",                  # 取消注释并修改为你显卡支持的batch size
    "--epochs", "12",                     # 网格搜索比较耗时，建议先用较少epoch(如12)看趋势
]

# 3. 遍历所有参数组合
for l1, l2, l3 in itertools.product(lambda1_vals, lambda2_vals, lambda3_vals):
    # 为每组实验创建一个独立的输出目录，防止权重和日志相互覆盖
    exp_name = f"l1_{l1}_l2_{l2}_l3_{l3}"
    output_dir = os.path.join("logs", "grid_search", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Starting experiment: {exp_name}")
    print(f"Parameters -> Lambda 1 (Coverage): {l1}, Lambda 2 (Spacing): {l2}, Lambda 3 (Count): {l3}")
    print(f"{'=' * 50}")

    # 组合当前实验的完整命令
    # 注意这里将 interval_loss_coef 和 ccm_loss_coef 设为同一个值，确保 L_count 权重一致
    current_cmd = base_cmd + [
        "--output_dir", output_dir,
        "--coverage_loss_coef", str(l1),
        "--spacing_loss_coef", str(l2),
        "--interval_loss_coef", str(l3),
        "--ccm_loss_coef", str(l3)
    ]

    # 执行命令 (会把日志输出到控制台，同时由于 DETR 会在 output_dir 中生成 log.txt)
    try:
        subprocess.run(current_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp_name} failed with error: {e}")
        # 如果中途某一组报错，是否选择继续下一组（直接 pass 即可）
        pass

print("\nGrid search completed! Check the 'logs/grid_search' directory for all results.")