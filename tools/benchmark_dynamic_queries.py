#!/usr/bin/env python
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import build_model_main
from util.checkpoint import load_native_resume
from util.slconfig import SLConfig
from util.config_validation import validate_config


def build_args(config_path, device):
    cfg = SLConfig.fromfile(config_path)._cfg_dict.to_dict()
    validate_config(cfg)
    cfg.update(device=device, distributed=False, use_ema=False)
    return argparse.Namespace(**cfg)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(model, images, warmup, iterations, device, amp=True):
    def forward():
        with torch.amp.autocast(device.type, enabled=amp and device.type == 'cuda'):
            return model(images)
    for _ in range(warmup):
        forward()
    synchronize(device)
    latencies = []
    counts = []
    executed_tokens = []
    padded_tokens = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iterations):
        start = time.perf_counter()
        outputs = forward()
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
        counts.extend(outputs['executed_query_counts'].detach().cpu().tolist())
        budget = outputs['allocator_outputs']
        executed_tokens.append(float(budget['decoder_query_tokens']))
        padded_tokens.append(float(budget['legacy_query_tokens']))
    peak = (torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == 'cuda' else 0.0)
    ordered = sorted(latencies)
    percentile = lambda p: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]
    return {
        'mean_latency_ms': statistics.fmean(latencies),
        'p50_latency_ms': percentile(0.50),
        'p90_latency_ms': percentile(0.90),
        'images_per_second': len(images) * 1000.0 / statistics.fmean(latencies),
        'peak_memory_mb': peak,
        'peak_reserved_mb': torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0.,
        'input_sizes': [list(image.shape[-2:]) for image in images],
        'amp': amp,
        'mean_query_count': statistics.fmean(counts),
        'decoder_query_tokens': statistics.fmean(executed_tokens),
        'query_token_reduction_percent':
            (1.0 - sum(executed_tokens) / max(sum(padded_tokens), 1.0)) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description='AQFC-DETR dynamic-query benchmark')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='efficiency_summary.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--width', type=int, default=1333)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--images', help='JSON list of {path, gt_count}; real images benchmark, batch=1')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.batch_size < 1:
        parser.error('Invalid warmup, iterations or batch size')
    if Path(args.output).exists():
        parser.error('Output already exists; choose a new --output')
    if args.images:
        from util.inference_runner import InferenceRunner
        from util.error_analysis import density_bucket
        from PIL import Image
        if args.batch_size != 1:
            parser.error('Real-image per-image/end-to-end benchmark currently requires batch-size=1')
        items = json.loads(Path(args.images).read_text(encoding='utf-8'))
        if not items:
            parser.error('Image manifest is empty')
        runner = InferenceRunner(args.config, args.checkpoint, args.device, args.amp)
        rows = []
        for mode, forced, grouped in [('fixed_900',900,False), ('batch_max_dynamic',None,False),
                                      ('grouped_dynamic',None,True)]:
            if runner.model.transformer.query_allocator is None and forced is None:
                continue
            runner.model.transformer.force_query_budget = forced
            runner.model.transformer.grouped_decoder_inference = grouped
            for item in items:
                # Image decode is outside end_to_end: preprocessing+H2D+model+postprocess are included.
                image = Image.open(item['path']).convert('RGB')
                for _ in range(args.warmup):
                    runner(image)
                runner.records.clear()
                if runner.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(runner.device)
                for _ in range(args.iterations):
                    runner(image)
                measured = list(runner.records)
                def summary(key):
                    values = sorted(r[key] for r in measured)
                    return dict(mean=statistics.fmean(values), p50=values[round((len(values)-1)*.5)],
                                p90=values[round((len(values)-1)*.9)])
                rows.append(dict(mode=mode, path=item['path'], density_bucket=density_bucket(item['gt_count']),
                    model_ms=summary('model_ms'), end_to_end_ms=summary('end_to_end_ms'),
                    mean_query_count=statistics.fmean(r['query_count'] for r in measured),
                    input_size=measured[0]['input_size'],
                    peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if runner.device.type=='cuda' else 0,
                    peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20 if runner.device.type=='cuda' else 0))
        groups = {}
        for row in rows:
            group = groups.setdefault(row['mode']+'/'+row['density_bucket'], [])
            group.append(row['end_to_end_ms']['mean'])
        result = dict(source='real_images', rows=rows,
                      density_group_mean_ms={k:statistics.fmean(v) for k,v in groups.items()},
                      timing_scope='end_to_end excludes file decode; includes preprocess/H2D/postprocess',
                      torch=torch.__version__, cuda=torch.version.cuda, amp=args.amp)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
        return
    device = torch.device(args.device)
    model, _, _ = build_model_main(build_args(args.config, args.device))
    load_native_resume(model, args.checkpoint)
    model.to(device).eval()
    images = [torch.rand(3, args.height, args.width, device=device)
              for _ in range(args.batch_size)]
    transformer = model.transformer
    modes = {
        'fixed_900': (900, False),
        'batch_max_dynamic': (None, False),
        'grouped_dynamic': (None, True),
    }
    results = {}
    for name, (forced, grouped) in modes.items():
        if transformer.query_allocator is None and forced is None:
            continue
        transformer.force_query_budget = forced
        transformer.grouped_decoder_inference = grouped
        results[name] = measure(model, images, args.warmup, args.iterations, device, args.amp)
    results['metadata'] = dict(source='synthetic', seed=torch.initial_seed(), torch=torch.__version__,
                               cuda=torch.version.cuda, warning='Not a natural-image query budget distribution')
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
