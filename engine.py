"""Training and evaluation loops for AQFC-DETR."""
import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import torch.nn as nn
import torch.nn.functional as F
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

from models.aqfcdetr.query_allocator import QueryBudgetLoss

print_freq = 5000


# Mosaic probability scheduler (epoch-aware warmup and decay).

class MosaicPScheduler:
    """
    Mosaic 触发概率的 epoch 级调度器。

    解决的问题：
        从已收敛的高精度 checkpoint 出发，在训练初期立刻以高概率
        触发 Mosaic，会把模型"打散"，需要很多 epoch 才能恢复。
        用 warmup + decay 策略可以让 Mosaic 在合适的时机发挥作用。

    三阶段调度：
        Warmup  (epoch 0 ~ warmup_end)      : 0 → peak_p，线性增长
        Plateau (epoch warmup_end ~ decay_start): peak_p，稳定
        Decay   (epoch decay_start ~ total) : peak_p → 0，余弦退出

    余弦退出的意义：
        训练末期用纯净样本让 EMA 收敛到更稳的极值，
        测试时 FP 会降低（与训练分布更接近）。

    使用示例（在 main.py 中）：
        mosaic_scheduler = MosaicPScheduler(
            peak_p=0.3,
            warmup_end=2,       # 前2个epoch warmup
            decay_start=14,     # 第14个epoch开始退出
            total_epochs=18,
        )
        # 在每个 epoch 开始时调用：
        dataset_train.mosaic.p = mosaic_scheduler.get_p(epoch)
    """

    def __init__(
        self,
        peak_p: float = 0.3,
        warmup_end: int = 2,
        decay_start: int = 8,
        total_epochs: int = 12,
    ):
        assert 0 < warmup_end < decay_start < total_epochs
        self.peak_p       = peak_p
        self.warmup_end   = warmup_end
        self.decay_start  = decay_start
        self.total_epochs = total_epochs

    def get_p(self, epoch: int) -> float:
        """返回当前 epoch 对应的 Mosaic 触发概率。"""
        if epoch < self.warmup_end:
            # 线性 warmup：0 → peak_p
            return self.peak_p * epoch / self.warmup_end
        elif epoch < self.decay_start:
            # 稳定阶段
            return self.peak_p
        else:
            # 余弦退出：peak_p → 0
            progress = (epoch - self.decay_start) / (self.total_epochs - self.decay_start)
            return self.peak_p * 0.5 * (1 + math.cos(math.pi * progress))

    def __repr__(self):
        return (
            f"MosaicPScheduler(peak_p={self.peak_p}, "
            f"warmup={self.warmup_end}, decay_start={self.decay_start}, "
            f"total={self.total_epochs})"
        )


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None,
                    logger=None, ema_m=None,
                    mosaic_scheduler=None):

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    if mosaic_scheduler is not None:
        dataset = data_loader.dataset
        # 兼容 DistributedSampler 包装
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if hasattr(dataset, 'mosaic') and dataset.mosaic is not None:
            new_p = mosaic_scheduler.get_p(epoch)
            dataset.mosaic.p = new_p
            if args.rank == 0:
                print(f"[MosaicScheduler] Epoch {epoch}: mosaic.p = {new_p:.3f}")

    budget_criterion = QueryBudgetLoss(
        coverage_weight=args.coverage_loss_weight,
        spacing_weight=args.spacing_loss_weight,
        count_weight=args.count_loss_weight,
        interval_weight=args.interval_loss_weight,
        boundary_guide_weight=args.boundary_guide_loss_weight,
        density_weight=args.density_map_loss_weight,
        enable_adaptive_targets=True,
        enable_loss_clipping=True
    ).to(device)
    budget_criterion.train()

    allocator_weight_scheduler = AllocatorWeightScheduler(
        warmup_epochs=args.allocator_schedule['warmup_epochs'],
        peak_weight=args.allocator_schedule['peak_weight'],
        final_weight=args.allocator_schedule['final_weight'],
        total_epochs=args.epochs if hasattr(args, 'epochs') else 24
    )

    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    if hasattr(model_without_ddp, 'set_epoch'):
        model_without_ddp.set_epoch(epoch)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('allocator_weight', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    if mosaic_scheduler is not None:
        metric_logger.add_meter('mosaic_p', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)

        real_counts = torch.tensor(
            [len(t['labels']) for t in targets],
            device=device,
            dtype=torch.float32
        )

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        current_allocator_weight = allocator_weight_scheduler.get_weight(epoch, _cnt)

        with torch.amp.autocast('cuda', enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            allocator_outputs = outputs.get('allocator_outputs')
            if allocator_outputs:
                budget_loss_out = budget_criterion(
                    allocator_outputs,
                    {'real_counts': real_counts, 'targets': targets})
                weighted_allocator_loss = (
                    budget_loss_out['loss_allocator_total'] *
                    current_allocator_weight * args.allocator_loss_weight)
                losses = losses + weighted_allocator_loss
                loss_dict['loss_allocator_total'] = weighted_allocator_loss
                for key in ('loss_budget_coverage', 'loss_budget_interval', 'loss_count',
                            'loss_budget_spacing', 'loss_boundary_guide', 'loss_density_map'):
                    loss_dict[key] = budget_loss_out[key]
                if _cnt % 1000 == 0 and args.rank == 0:
                    query_counts = outputs['executed_query_counts'].float()
                    message = (
                        f"[AQBA] epoch={epoch} iter={_cnt} "
                        f"teacher={allocator_outputs['teacher_ratio']:.2f} "
                        f"weight={current_allocator_weight:.2f} "
                        f"queries={query_counts.mean().item():.1f} "
                        f"count_mae={(allocator_outputs['predicted_count'] - real_counts).abs().mean().item():.2f} "
                        f"loss={weighted_allocator_loss.item():.4f}")
                    print(message)
                    if logger:
                        logger.info(message)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}

        weight_dict['loss_allocator_total'] = 1.0

        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(allocator_weight=current_allocator_weight)
        if allocator_outputs:
            executed = outputs['executed_query_counts'].float()
            predicted = allocator_outputs['predicted_count'].float()
            metric_logger.update(
                teacher_ratio=float(allocator_outputs['teacher_ratio']),
                mean_gt_count=real_counts.mean().item(),
                mean_predicted_count=predicted.mean().item(),
                count_mae=(predicted - real_counts).abs().mean().item(),
                mean_executed_queries=executed.mean().item(),
                invalid_allocator_fallback_count=float(
                    allocator_outputs.get('invalid_fallback_count', 0)),
                query_token_reduction=float(
                    allocator_outputs.get('query_token_reduction', 0.0)))
            for level in args.query_budget_levels:
                metric_logger.update(**{
                    f'query_level_{level}_ratio': (executed == level).float().mean().item()})
        if mosaic_scheduler is not None:
            cur_dataset = data_loader.dataset
            if hasattr(cur_dataset, 'dataset'):
                cur_dataset = cur_dataset.dataset
            if hasattr(cur_dataset, 'mosaic') and cur_dataset.mosaic is not None:
                metric_logger.update(mosaic_p=cur_dataset.mosaic.p)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


# ======================================================================
# AQBA auxiliary-loss scheduler
# ======================================================================

class AllocatorWeightScheduler:
    """Warm up and gradually decay the AQBA auxiliary-loss weight."""
    def __init__(self, warmup_epochs=3, peak_weight=1.0, final_weight=0.7, total_epochs=24):
        self.warmup_epochs = warmup_epochs
        self.peak_weight   = peak_weight
        self.final_weight  = final_weight
        self.total_epochs  = total_epochs

    def get_weight(self, epoch: int, iteration: int) -> float:
        if epoch < self.warmup_epochs:
            return self.peak_weight * (epoch + iteration / 14018) / self.warmup_epochs
        elif epoch < int(self.total_epochs * 0.75):
            return self.peak_weight
        else:
            decay_progress = (epoch - int(self.total_epochs * 0.75)) / (
                self.total_epochs - int(self.total_epochs * 0.75)
            )
            return self.peak_weight - (self.peak_weight - self.final_weight) * decay_progress


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    evaluated_query_counts = []
    output_state_dict = {}
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}

        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled
        )
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'executed_query_counts' in outputs:
            query_counts = outputs['executed_query_counts'].float()
            evaluated_query_counts.extend(query_counts.cpu().tolist())
            metric_logger.update(mean_executed_queries=query_counts.mean().item())
            allocator_outputs = outputs.get('allocator_outputs', {})
            metric_logger.update(
                decoder_query_tokens=float(allocator_outputs.get(
                    'decoder_query_tokens', query_counts.sum())),
                query_token_reduction=float(allocator_outputs.get(
                    'query_token_reduction', 0.0)))

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                gt_bbox  = tgt['boxes']
                gt_label = tgt['labels']
                gt_info  = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                _res_bbox  = outbbox
                _res_prob  = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                if 'gt_info'  not in output_state_dict: output_state_dict['gt_info']  = []
                if 'res_info' not in output_state_dict: output_state_dict['res_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if evaluated_query_counts:
        ordered_counts = sorted(evaluated_query_counts)
        stats['p50_query_count'] = ordered_counts[round((len(ordered_counts) - 1) * 0.50)]
        stats['p90_query_count'] = ordered_counts[round((len(ordered_counts) - 1) * 0.90)]
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th']  = panoptic_res["Things"]
        stats['PQ_st']  = panoptic_res["Stuff"]

    return stats, coco_evaluator
