import torch
import torch.nn.functional as F

# 加载模型
ckpt = torch.load('/workspace/DQDetr/logs/DQDETR_ver1/0102完整训练/checkpoint.pth', map_location='cpu')
state_dict = ckpt['model']

# 查看边界头的权重和偏置
b_weight = state_dict['transformer.CCM.boundary_head.4.weight']
b_bias = state_dict['transformer.CCM.boundary_head.4.bias']

print("=" * 60)
print("【原始参数】")
print("Weight shape:", b_weight.shape)
print("Bias (raw):", b_bias)
print()

# 【正确的计算方式】模拟前向传播
# 假设输入特征经过 boundary_head 前面的层后变为 0（用于观察初始偏置的影响）
raw_out = b_bias  # 模拟 Linear(input=0) + bias 的结果

# 按照 ccm.py 中的逻辑计算
log_b1 = raw_out[0].clamp(min=1.0, max=8.0)

min_log_gap = 0.2
delta12 = F.softplus(raw_out[1]) + min_log_gap
delta23 = F.softplus(raw_out[2]) + min_log_gap

log_b2 = log_b1 + delta12
log_b3 = log_b2 + delta23

log_boundaries = torch.stack([log_b1, log_b2, log_b3])
boundaries = torch.exp(log_boundaries)

print("【正确的边界计算】")
print(f"log_b1 = clamp({raw_out[0]:.4f}) = {log_b1:.4f}")
print(f"delta12 = softplus({raw_out[1]:.4f}) + {min_log_gap} = {delta12:.4f}")
print(f"delta23 = softplus({raw_out[2]:.4f}) + {min_log_gap} = {delta23:.4f}")
print()
print(f"log_boundaries: {log_boundaries}")
print(f"实际边界值 (b1, b2, b3): {boundaries}")
print()

# 验证边界是否合理
print("【边界合理性检查】")
print(f"b1 = {boundaries[0]:.2f} (期望约 20-50)")
print(f"b2 = {boundaries[1]:.2f} (期望约 50-200)")
print(f"b3 = {boundaries[2]:.2f} (期望约 200-800)")
print(f"b2/b1 比例: {boundaries[1]/boundaries[0]:.2f}")
print(f"b3/b2 比例: {boundaries[2]/boundaries[1]:.2f}")
print()

# 计算覆盖率（假设数据集分布）
# 假设 log_count 服从正态分布 N(4.5, 1.0)，即 count ≈ exp(4.5) ≈ 90
test_log_counts = torch.linspace(2.0, 7.0, 100)  # log(7) ≈ 2, log(1000) ≈ 7

tau = 1.0
cdf_b1 = torch.sigmoid((log_boundaries[0] - test_log_counts) / tau)
cdf_b2 = torch.sigmoid((log_boundaries[1] - test_log_counts) / tau)
cdf_b3 = torch.sigmoid((log_boundaries[2] - test_log_counts) / tau)

print("【理论覆盖率】(均值 across log_count ∈ [2, 7])")
print(f"P(count < b1): {cdf_b1.mean():.2%}")
print(f"P(count < b2): {cdf_b2.mean():.2%}")
print(f"P(count < b3): {cdf_b3.mean():.2%}")
print()

# 给出查询数量建议
print("【Query档位映射示例】")
query_levels = [300, 500, 900, 1500]
for i, (lower, upper) in enumerate([(0, boundaries[0]),
                                      (boundaries[0], boundaries[1]),
                                      (boundaries[1], boundaries[2]),
                                      (boundaries[2], float('inf'))]):
    print(f"档位 {i}: [{lower:.1f}, {upper:.1f}) → {query_levels[i]} queries")

print("=" * 60)