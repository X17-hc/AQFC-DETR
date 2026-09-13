 # Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import gc
import json
import random
import time
from pathlib import Path
import os, sys
if sys.platform != 'win32':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma
from util.runtime import run_metadata, can_select_best
from util.experiment import unique_output, write_manifest, variant_signature
from util.checkpoint import load_native_resume, migration_report_path, capture_rng_state
from util.checkpoint_migration import load_legacy_pretrained
from util.config_validation import validate_config
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, MosaicPScheduler

# 默认项目内路径以 main.py 所在目录为基准，不依赖 PyCharm 的工作目录。
PROJECT_ROOT = Path(__file__).resolve().parent


class ExplicitPretrainedAction(argparse.Action):
    """记录用户是否显式传入预训练参数，区分它与 default 自动填入的值。"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.pretrained_explicit = True


def resolve_launch_defaults(args):
    if getattr(args, 'eval_ema', False) and not (args.eval and args.resume):
        raise ValueError('--eval-ema requires --eval and --resume with an EMA checkpoint')
    # 续训恢复完整 checkpoint，不能再次加载默认旧权重覆盖恢复结果。
    # 不能仅比较文件名：用户也可能显式传入与默认值完全相同的权重路径。
    explicit = getattr(args, 'pretrained_explicit', False)
    if getattr(args, 'no_pretrained', False):
        if explicit:
            raise ValueError('--pretrained and --no-pretrained cannot be used together')
        args.pretrain_model_path = ''
    elif args.resume:
        if explicit and args.pretrain_model_path:
            raise ValueError('Specify only one of --resume and --pretrained')
        args.pretrain_model_path = ''
    return args


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # 本机 8GB 显存训练配置；不传 --config 时使用此文件，命令行可覆盖。
    parser.add_argument('--config', '-c', dest='config_file', type=str,
                        default=str(PROJECT_ROOT / 'configs/aitodv2/aqfc_r50_5scale_local8gb.py'))
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    # parser.add_argument('--dataset_file', default='visdrone')
    parser.add_argument('--dataset_file', default=None)
    # parser.add_argument('--dataset_file', default='aitod')
    # parser.add_argument('--dataset_file', default='coco')

    # 直接读取已有数据集，不复制数据；换机器时可修改默认值或传 --data-root。
    parser.add_argument('--data-root', dest='coco_path', type=str,
                        default='D:/PythonProject/DQ-DETR/DQ-DETR/data/path/AITODv2')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    # 默认训练日志和 checkpoint 保存位置；不同实验建议指定不同输出目录。
    parser.add_argument('--output-dir', dest='output_dir',
                        default=str(PROJECT_ROOT / 'outputs/aitodv2_pycharm'),
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--unique-output-dir', action='store_true',
                        help='Create a timestamped child for a new experiment; never resume')
    parser.add_argument('--export-predictions', action='store_true',
                        help='Export unthresholded COCO predictions and per-image metadata')
    parser.add_argument('--eval-image-ids', default='',
                        help='Evaluation-only JSON list of unique image IDs; empty uses the full split')
    parser.add_argument('--eval-query-floor', type=int, default=0,
                        help='Evaluation-only minimum query budget; zero disables; preserves higher budgets')
    parser.add_argument('--max-consecutive-skipped-steps', type=int, default=0,
                        help='Opt-in stop after this many consecutive optimizer skips; zero disables')
    parser.add_argument('--export-diagnostics', action='store_true',
                        help='With prediction export, save density/proposal arrays for at most 32 images per rank')
    parser.add_argument('--profile-trace', default='',
                        help='Opt-in bounded training Chrome trace, relative to output directory')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 新训练默认加载旧最佳权重。--resume 会取消此默认值；显式同时指定则报错。
    parser.add_argument('--pretrained', dest='pretrain_model_path', type=str,
                        default=str(PROJECT_ROOT / 'weights/legacy/dqdetr_best305.pth'),
                        action=ExplicitPretrainedAction,
                        help='override the local default warm-start checkpoint')
    # 禁用检测模型的默认 warm-start；不改变 backbone 原有的 ImageNet 初始化。
    parser.add_argument('--no-pretrained', action='store_true', default=False,
                        help='disable default warm-start weights')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-ema', action='store_true',
                        help='Evaluate ema_model instead of model; requires --eval --resume')
    # Windows/PyCharm 默认使用主进程读取数据，避免多进程启动和额外内存开销。
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction,
                        help="Train with mixed precision", default=True)
    parser.add_argument('--max-train-steps', type=int, default=0,
                        help='Limit steps per epoch for smoke tests; 0 means full epoch')
    parser.add_argument('--max-eval-steps', type=int, default=0,
                        help='Limit validation batches; 0 means full evaluation')

    return parser


def build_data_loaders(args):
    """Evaluation needs only the selected evaluation split, never train images."""
    args.train_split = getattr(args, 'train_split', 'trainval')
    args.eval_split = getattr(args, 'eval_split', 'test')
    dataset_val = build_dataset(image_set=args.eval_split, args=args)
    if getattr(args, 'eval_image_ids', ''):
        if not args.eval:
            raise ValueError('--eval-image-ids is evaluation-only')
        from util.factor_diagnostics import restrict_eval_dataset
        restrict_eval_dataset(dataset_val, args.eval_image_ids)
    sampler_val = (DistributedSampler(dataset_val, shuffle=False) if args.distributed
                   else torch.utils.data.SequentialSampler(dataset_val))
    loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False,
                            collate_fn=utils.collate_fn, num_workers=args.num_workers)
    if args.eval:
        return None, dataset_val, None, loader_val, None
    dataset_train = build_dataset(image_set=args.train_split, args=args)
    sampler_train = (DistributedSampler(dataset_train) if args.distributed
                     else torch.utils.data.RandomSampler(dataset_train))
    batches = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    loader_train = DataLoader(dataset_train, batch_sampler=batches, collate_fn=utils.collate_fn,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    if len(loader_train) == 0:
        raise ValueError('Training loader is empty; check dataset size and batch_size')
    return dataset_train, dataset_val, loader_train, loader_val, sampler_train


def build_model_main(args):
    if args.modelname != 'aqfcdetr':
        raise ValueError(f"Unsupported modelname: {args.modelname}")
    from models.aqfcdetr import build_aqfcdetr
    return build_aqfcdetr(args)

def main(args):
    args = resolve_launch_defaults(args)
    if getattr(args, 'unique_output_dir', False) and getattr(args, 'world_size', 1) > 1:
        raise ValueError('Unique output directories currently require a single-process launch')
    unique_output(args)
    if not args.resume and not args.eval and any(Path(args.output_dir).glob('checkpoint*.pth')):
        raise FileExistsError('Output directory already contains checkpoints; use a new --output-dir or --resume')
    if min(args.max_train_steps, args.max_eval_steps) < 0:
        raise ValueError('Step limits must be non-negative')
    if args.profile_trace and not 1 <= args.max_train_steps <= 100:
        raise ValueError('Profiling requires --max-train-steps in [1,100]')
    if args.export_diagnostics and not args.export_predictions:
        raise ValueError('--export-diagnostics requires --export-predictions')
    # 定期清理显存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    utils.init_distributed_mode(args)
    # load cfg file and update the args
    # print("Loading config file from {}".format(args.config_file))

    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    validate_config(cfg_dict)
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars or args_vars[k] is None:
            setattr(args, k, v)
        else:
            print("ex:", k, v)
            raise ValueError("Key {} can used by args only".format(k))

    from util.runtime import validate_eval_query_floor
    validate_eval_query_floor(args.eval_query_floor, args.eval, args.query_budget_levels,
                              getattr(args, 'force_query_budget', None))
    if args.max_consecutive_skipped_steps < 0:
        raise ValueError('--max-consecutive-skipped-steps must be nonnegative')
    if getattr(args, 'run_purpose', 'engineering_check') == 'research' and args.pretrain_model_path:
        raise ValueError('Research configurations require --no-pretrained; audit new initialization separately')

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info(f'Model={args.modelname}; config={args.config_file}; device={args.device}; '
                f'output={args.output_dir}. Full arguments are in config_args_all.json.')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    model.transformer.eval_query_floor = args.eval_query_floor
    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    # DDP broadcasts initial parameters: copy EMA only after that synchronization.
    ema_m = (ModelEma(model_without_ddp, args.ema_decay)
             if (args.use_ema and not args.eval) or args.eval_ema else None)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    if args.rank == 0:
        with (Path(args.output_dir) / 'parameter_counts.json').open('w') as handle:
            json.dump({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, handle, indent=2)

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp,
                                init_scale=getattr(args, 'amp_init_scale', 128.))

    dataset_train, dataset_val, data_loader_train, data_loader_val, sampler_train = build_data_loaders(args)
    if not can_select_best(args):
        logger.warning('Best-checkpoint selection disabled: test split or smoke/partial run. '
                       'Use train/val splits and full validation for model selection.')
    if args.eval:
        lr_scheduler = None
    elif args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                           steps_per_epoch=len(data_loader_train), epochs=args.epochs,
                                                           pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    best_metrics = {}
    if args.resume and args.pretrain_model_path:
        raise ValueError('Specify only one of --resume and --pretrained')
    if args.resume:
        args.start_epoch = load_native_resume(
            model_without_ddp, args.resume,
            optimizer=None if args.eval else optimizer,
            scheduler=None if args.eval else lr_scheduler,
            ema=ema_m, scaler=None if args.eval else scaler,
            best_metrics=best_metrics, expected_args=args)

    if (not args.resume) and args.pretrain_model_path:
        report = load_legacy_pretrained(
            model_without_ddp, args.pretrain_model_path,
            report_path=migration_report_path(args.output_dir))
        logger.info(
            f"Warm-start coverage: {report['coverage_by_numel']:.2%}; "
            f"missing={len(report['missing_keys'])}, mismatched={len(report['shape_mismatched_keys'])}")
        if ema_m is not None:
            ema_m.module.load_state_dict(model_without_ddp.state_dict(), strict=True)

    if args.rank == 0:
        write_manifest(args, PROJECT_ROOT)
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        evaluation_model = ema_m.module if args.eval_ema else model
        logger.info('Evaluation weights: %s', 'ema_model' if args.eval_ema else 'model')
        test_stats, coco_evaluator = evaluate(evaluation_model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir,
                                              wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     'evaluation_weights': 'ema_model' if args.eval_ema else 'model'}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with (output_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            query_stats = {k: v for k, v in test_stats.items()
                           if 'query' in k or 'decoder' in k}
            with (output_dir / "query_budget_statistics.json").open("w") as f:
                json.dump(query_stats, f, indent=2)

        return

    start_time = time.time()

    # 新增 MosaicPScheduler 构建
    mosaic_scheduler = None
    _ds = dataset_train
    if hasattr(_ds, 'dataset'):
        _ds = _ds.dataset
    if hasattr(_ds, 'mosaic') and _ds.mosaic is not None:
        mosaic_scheduler = MosaicPScheduler(
            peak_p=getattr(args, 'mosaic_p', 0.3),
            warmup_end=getattr(args, 'mosaic_warmup_end', 2),
            decay_start=getattr(args, 'mosaic_decay_start', 8),  # 原14，为了配合SBA的超参数实验，改成了8
            total_epochs=args.epochs,
        )
        if args.rank == 0:
            print(f"[MosaicPScheduler] {mosaic_scheduler}")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        best_checkpoint_paths = []
        if args.distributed:
            sampler_train.set_epoch(epoch)
        from util.profiling import training_profile
        trace = str(output_dir / f'epoch_{epoch}_{Path(args.profile_trace).name}') if args.profile_trace else ''
        with training_profile(trace, model, criterion):
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
                logger=(logger if args.save_log else None), ema_m=ema_m,
                mosaic_scheduler=mosaic_scheduler, scaler=scaler)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if not args.max_train_steps and ((epoch + 1) % args.lr_drop == 0 or
                                             (epoch + 1) % args.save_checkpoint_interval == 0):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'scaler': scaler.state_dict(),
                    'format': 'aqfcdetr_v2',
                    'variant_signature': variant_signature(args),
                    'run_metadata': run_metadata(args, train_stats['train_iterations'], len(data_loader_train)),
                    'best_metrics': best_metrics,
                }
            if args.rank == 0:
                write_manifest(args, PROJECT_ROOT, update={
                    'last_epoch': epoch, **weights['run_metadata']})
            if args.use_ema:
                weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })

        # eval
        if epoch in args.val_epoch:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = can_select_best(args) and map_regular >= 0 and map_regular > best_metrics.get('regular_AP', -1.)
            if _isbest and args.output_dir:
                best_metrics.update(regular_AP=map_regular, regular_epoch=epoch)
                best_checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }

            # eval ema
            if args.use_ema:
                ema_test_stats, ema_coco_evaluator = evaluate(
                    ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
                )
                log_stats.update({f'ema_test_{k}': v for k, v in ema_test_stats.items()})
                map_ema = ema_test_stats['coco_eval_bbox'][0]
                _isbest = can_select_best(args) and map_ema >= 0 and map_ema > best_metrics.get('ema_AP', -1.)
                if _isbest and args.output_dir:
                    best_metrics.update(ema_AP=map_ema, ema_epoch=epoch)
                    best_checkpoint_paths.append(output_dir / 'checkpoint_best_ema.pth')

            log_stats.update(best_metrics=best_metrics)

            ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass

            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                with (output_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)
        elif args.output_dir and utils.is_main_process():
            log_stats = {f'train_{k}': v for k, v in train_stats.items()}
            log_stats.update(epoch=epoch, n_parameters=n_parameters)
            for log_name in ('log.txt', 'metrics.jsonl'):
                with (output_dir / log_name).open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')
        # Save after validation so resume retains the latest selection history.
        if args.output_dir:
            # Collect each rank's post-validation RNG stream for epoch-boundary resume.
            weights['rng_states'] = utils.all_gather(capture_rng_state())
            for checkpoint_path in checkpoint_paths + best_checkpoint_paths:
                utils.save_on_master(weights, checkpoint_path)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
