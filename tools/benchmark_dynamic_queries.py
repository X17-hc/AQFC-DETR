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
def measure(model, images, warmup, iterations, device):
    for _ in range(warmup):
        model(images)
    synchronize(device)
    latencies = []
    counts = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iterations):
        start = time.perf_counter()
        outputs = model(images)
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
        counts.extend(outputs['executed_query_counts'].detach().cpu().tolist())
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
        'mean_query_count': statistics.fmean(counts),
        'decoder_query_tokens': sum(counts) / iterations,
        'query_token_reduction_percent':
            (1.0 - statistics.fmean(counts) / max(counts)) * 100.0,
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
    args = parser.parse_args()
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
        transformer.force_query_budget = forced
        transformer.grouped_decoder_inference = grouped
        results[name] = measure(model, images, args.warmup, args.iterations, device)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
