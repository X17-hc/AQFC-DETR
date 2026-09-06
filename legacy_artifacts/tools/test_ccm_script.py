"""
测试CCM自适应边界机制
"""
import torch
import sys
sys.path.append('.')

from models.dqdetr.ccm import AdaptiveBoundaryCCM, TrueAdaptiveBoundaryLoss
from util.box_ops import (
    validate_boundary_predictions, 
    compute_box_size_distribution,
    assign_boxes_to_boundaries,
    compute_boundary_coverage
)

def test_ccm_forward():
    """测试CCM前向传播"""
    print("\n" + "="*60)
    print("Test 1: CCM Forward Pass")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建CCM模块
    ccm = AdaptiveBoundaryCCM(
        feature_dim=256,
        ccm_cls_num=4,
        query_levels=[300, 500, 900, 1500],
        max_objects=1500,
        use_soft_assignment=True
    ).to(device)
    
    # 模拟特征图
    batch_size = 2
    h, w = 50, 50
    feature_map = torch.randn(batch_size, 256, h, w).to(device)
    
    # 模拟真实计数
    real_counts = torch.tensor([50, 200], device=device, dtype=torch.float32)
    
    # 前向传播
    try:
        outputs = ccm(feature_map, real_counts=real_counts)
        
        print("✓ Forward pass successful")
        print(f"  Boundaries: {outputs['pred_boundaries']}")
        print(f"  Predicted count: {outputs['predicted_count']}")
        print(f"  Num queries: {outputs['num_queries']}")
        
        # 验证边界
        valid = validate_boundary_predictions(
            outputs['pred_boundaries'],
            outputs['log_boundaries']
        )
        print(f"  Valid boundaries: {valid}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_adaptive_loss():
    """测试自适应损失"""
    print("\n" + "=" * 60)
    print("Test 2: Adaptive Boundary Loss")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建损失函数
    criterion = TrueAdaptiveBoundaryLoss(
        coverage_weight=20.0,
        spacing_weight=1.0,
        count_weight=1.0,
        interval_weight=2.0
    ).to(device)

    # 模拟CCM输出 (关键修改：模拟神经网络输出，开启 requires_grad)
    batch_size = 2

    # 1. log_boundaries (作为叶子节点，需要梯度)
    log_boundaries_data = torch.log(torch.tensor([[20., 100., 500.], [30., 150., 600.]], device=device))
    log_boundaries = log_boundaries_data.clone().detach().requires_grad_(True)

    # 2. raw_count (作为叶子节点，需要梯度)
    raw_count_data = torch.log(torch.tensor([100., 200.], device=device))
    raw_count = raw_count_data.clone().detach().requires_grad_(True)

    # 3. pred_bbox_number (分类logits，需要梯度)
    pred_bbox_number = torch.randn(batch_size, 4, device=device, requires_grad=True)

    # 构造输出字典
    # 注意：pred_boundaries 和 predicted_count 应该由带有梯度的变量计算得出，保持计算图连通
    outputs = {
        'pred_boundaries': torch.exp(log_boundaries),  # 保持梯度传递
        'log_boundaries': log_boundaries,
        'predicted_count': torch.exp(raw_count),  # 保持梯度传递
        'raw_count': raw_count,
        'pred_bbox_number': pred_bbox_number
    }

    # 真实计数 (Target不需要梯度)
    real_counts = torch.tensor([80, 180], device=device)

    try:
        loss_dict = criterion(outputs, {'real_counts': real_counts})

        print("✓ Loss computation successful")
        print(f"  Total loss: {loss_dict['total_adaptive_loss'].item():.4f}")
        print(f"  Coverage loss: {loss_dict['loss_coverage'].item():.4f}")
        print(f"  Interval loss: {loss_dict['loss_interval'].item():.4f}")
        print(f"  Count loss: {loss_dict['loss_count'].item():.4f}")
        print(f"  Coverage rates: {loss_dict['coverage_rates'].detach().cpu().numpy()}")

        # 测试反向传播
        loss_dict['total_adaptive_loss'].backward()

        # 验证梯度是否生成
        if log_boundaries.grad is not None and raw_count.grad is not None:
            print("✓ Backward pass successful (Gradients computed)")
            return True
        else:
            print("✗ Backward pass failed (No gradients found)")
            return False

    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_box_assignment():
    """测试边界框分配"""
    print("\n" + "="*60)
    print("Test 3: Box Assignment to Boundaries")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟边界框（归一化坐标）
    # 小框: 2x2 = 4 pixels
    # 中框: 10x10 = 100 pixels
    # 大框: 25x25 = 625 pixels
    boxes = torch.tensor([
        [0.25, 0.25, 0.255, 0.255],  # 4 pixels at 800x800
        [0.25, 0.50, 0.2625, 0.5125],  # 100 pixels
        [0.50, 0.50, 0.53125, 0.53125],  # 625 pixels
    ], device=device)
    
    boundaries = torch.tensor([20, 100, 500], device=device)
    
    try:
        assignments = assign_boxes_to_boundaries(boxes, boundaries)
        coverage = compute_boundary_coverage(boxes, boundaries)
        
        print("✓ Box assignment successful")
        print(f"  Assignments: {assignments}")
        print(f"  Coverage: {coverage}")
        
        return True
    except Exception as e:
        print(f"✗ Box assignment failed: {e}")
        return False


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n" + "="*60)
    print("Test 4: Numerical Stability")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ccm = AdaptiveBoundaryCCM(
        feature_dim=256,
        ccm_cls_num=4,
        query_levels=[300, 500, 900, 1500]
    ).to(device)
    
    # 极端情况：非常小的计数
    feature_map = torch.randn(1, 256, 50, 50).to(device)
    real_counts = torch.tensor([1], device=device, dtype=torch.float32)
    
    try:
        outputs = ccm(feature_map, real_counts=real_counts)
        
        # 检查是否有NaN/Inf
        has_nan = any(
            torch.isnan(v).any() if isinstance(v, torch.Tensor) else False
            for v in outputs.values()
        )
        has_inf = any(
            torch.isinf(v).any() if isinstance(v, torch.Tensor) else False
            for v in outputs.values()
        )
        
        if has_nan or has_inf:
            print(f"✗ Numerical instability detected: NaN={has_nan}, Inf={has_inf}")
            return False
        else:
            print("✓ Numerically stable")
            return True
            
    except Exception as e:
        print(f"✗ Stability test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("CCM Adaptive Boundary Mechanism Tests")
    print("="*60)
    
    results = {
        'forward': test_ccm_forward(),
        'loss': test_adaptive_loss(),
        'assignment': test_box_assignment(),
        'stability': test_numerical_stability()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
