import os
import sys
import torch
import argparse

# 导入 main_aitod 中的解析器和构建函数
from main_aitod import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list


def main():
    # 1. 借用原有的参数解析器
    parser = argparse.ArgumentParser('Compute FLOPs and Params', parents=[get_args_parser()])
    args, _ = parser.parse_known_args()  # 使用 parse_known_args 防止多余参数报错

    # 2. 解析配置文件
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()

    # 正确的合并方式：保留 args 中已有的命令行属性，只补充 cfg_dict 中新增的属性
    for k, v in cfg_dict.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # 手动补齐部分必须的基础属性(以防万一)
    if not hasattr(args, 'device'):
        args.device = 'cuda'
    if getattr(args, 'use_ema', None) is None:
        args.use_ema = False

    print("=> 正在构建你的模型 (Building model)...")
    # 3. 传入修复后的 args
    model, criterion, postprocessors = build_model_main(args)
    model.eval()  # 切换到推理模式
    model.to(args.device)  # 将模型移至对应的 device

    # =======================================================
    # 模块 A: 精确计算 Params
    # =======================================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "★" * 50)
    print(f"[Params 指标]")
    print(f"总参数量 (Total Params):     {total_params / 1e6:.3f} M")
    print(f"可训练参数 (Trainable):      {trainable_params / 1e6:.3f} M")
    print("★" * 50 + "\n")


    print("=> 正在构造带 Padding Mask 的真实 NestedTensor 输入...")
    # 模拟一张 800x800 的测试图像
    dummy_image = torch.randn(3, 800, 800).to(args.device)
    inputs = nested_tensor_from_tensor_list([dummy_image])

    print("\n=> 正在启用 PyTorch 官方底层 ATen 运算级 FLOPs 追踪 (最精准，无视 JIT)...")
    try:
        from torch.utils.flop_counter import FlopCounterMode

        # 拦截所有底层 ATen 操作
        flop_counter = FlopCounterMode(model, display=False)
        with flop_counter:
            _ = model(inputs)

        # PyTorch 原生统计的是纯 FLOPs (Floating Point Operations)
        # CV 论文中常说的 GFLOPs 实际上是 GMACs (乘加累积操作)
        # 换算公式: 1 MAC = 2 FLOPs
        total_flops = flop_counter.get_total_flops()
        gmacs = (total_flops / 2) / 1e9

        print("\n" + "★" * 60)
        print(f"[最终极、最精确的计算量指标]")
        print(f"底层浮点运算总数 (Raw FLOPs): {total_flops / 1e9:.3f} G")
        print(f"论文标准计算量 (GMACs / GFLOPs): {gmacs:.3f} G ！")
        print("★" * 60 + "\n")

    except ImportError:
        print("\n[错误] 当前 PyTorch 版本过低，无法使用 FlopCounterMode。")
        print("因为环境限制，推荐在论文中直接使用推理时间 (0.712 s/it) 作为效率论证，放弃静态 FLOPs。")


if __name__ == '__main__':
    main()