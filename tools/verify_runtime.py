"""Real CUDA model/criterion/allocator forward and backward smoke test."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.aqfcdetr import build_aqfcdetr
from models.aqfcdetr.query_allocator import QueryBudgetLoss
from util.slconfig import SLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/aitodv2/aqfc_r50_5scale_local8gb.py')
    parser.add_argument('--output', default='runtime_verification.json')
    parser.add_argument('--epoch', type=int, default=6, help='Default tests fully predicted routing')
    parser.add_argument('--no-dn', action='store_true', help='Exercise teacher routing independently of DN')
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config)._cfg_dict.to_dict()
    cfg.update(device='cuda')
    if args.no_dn:
        cfg['use_dn'] = False
    torch.manual_seed(42)
    model, criterion, post = build_aqfcdetr(argparse.Namespace(**cfg))
    model.cuda().train()
    criterion.cuda().train()
    model.set_epoch(args.epoch)
    # Mixed odd sizes exercise real padding masks instead of only square batches.
    image_sizes = [(129, 131), (137, 149)]
    images = [torch.rand(3, h, w, device='cuda') for h, w in image_sizes]
    targets = [{'labels': torch.tensor([1, 2], device='cuda'),
                'boxes': torch.tensor([[.3,.4,.03,.04],[.7,.6,.02,.03]], device='cuda'),
                'size': torch.tensor([h, w], device='cuda')} for h, w in image_sizes]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    with torch.autocast('cuda'):
        outputs = model(images, targets)
        losses = criterion(outputs, targets)
        allocation = QueryBudgetLoss().cuda()(outputs['allocator_outputs'], {
            'real_counts': torch.tensor([2.,2.], device='cuda'), 'targets': targets})
        loss = sum(v * criterion.weight_dict[k] for k,v in losses.items()
                   if k in criterion.weight_dict) + allocation['loss_allocator_total']
    assert torch.isfinite(loss), loss
    loss.backward()
    gradients = {}
    for name, module in [('allocator', model.transformer.query_allocator),
                         ('boundary_head', model.transformer.query_allocator.boundary_head),
                         ('density_head', model.transformer.query_allocator.density_head),
                         ('calibrator', model.transformer.feature_calibrator),
                         ('decoder', model.transformer.decoder)]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        gradients[name] = sum(g.abs().sum().item() for g in grads)
        assert gradients[name] > 0, name
    optimizer.step()
    del outputs, losses, allocation, loss
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        model.transformer.force_query_budget = 300
        model.transformer.grouped_decoder_inference = True
        grouped = model(images)
        model.transformer.grouped_decoder_inference = False
        padded = model(images)
        torch.testing.assert_close(grouped['pred_logits'], padded['pred_logits'], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(grouped['pred_boxes'], padded['pred_boxes'], atol=1e-5, rtol=1e-4)
        results = post['bbox'](grouped, torch.tensor(image_sizes, device='cuda'))
        assert all(len(r['scores']) == 300 for r in results)
        model.transformer.force_query_budget = None
        budgets = torch.tensor([300,500,900,1500], device='cuda')
        def force_mixed_budgets(module, inputs, result):
            result['query_counts'] = budgets
            return result
        hook = model.transformer.query_allocator.register_forward_hook(force_mixed_budgets)
        mixed_sizes = [(160, 160), (151, 157), (137, 149), (159, 153)]
        mixed_images = [torch.rand(3,h,w,device='cuda') for h, w in mixed_sizes]
        try:
            model.transformer.grouped_decoder_inference = True
            mixed_grouped = model(mixed_images)
            model.transformer.grouped_decoder_inference = False
            mixed_padded = model(mixed_images)
            mask = mixed_grouped['query_valid_mask']
            assert mask.sum(1).tolist() == budgets.tolist()
            torch.testing.assert_close(mixed_grouped['pred_logits'][mask],
                                       mixed_padded['pred_logits'][mask], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(mixed_grouped['pred_boxes'][mask],
                                       mixed_padded['pred_boxes'][mask], atol=1e-5, rtol=1e-4)
        finally:
            hook.remove()
    report = {'status':'passed', 'gradient_l1':gradients,
              'epoch': args.epoch, 'use_dn': cfg['use_dn'],
              'training_image_sizes': image_sizes, 'mixed_image_sizes': mixed_sizes,
              'grouped_equal_to_padded':True,
              'mixed_grouped_equal_to_padded':True,
              'mixed_query_counts': budgets.tolist(),
              'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
