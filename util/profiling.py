"""Opt-in trace ranges; normal training does not activate hooks or CUDA sync."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import time
import torch

_enabled = False


def range_summary(events):
    """Keep CPU dispatch and GPU annotations separate; never add nested ranges."""
    rows = []
    for event in events:
        if not event.key.startswith('AQFC/'):
            continue
        device_type = str(event.device_type).split('.')[-1]
        rows.append(dict(name=event.key, device_type=device_type,
            cpu_total_us=event.cpu_time_total, device_total_us=event.device_time_total,
            self_cpu_us=event.self_cpu_time_total,
            self_device_us=event.self_device_time_total, count=event.count,
            timing_semantics=('CPU inclusive dispatch/wait; device time is attributed work'
                              if device_type == 'CPU' else
                              'Device annotation span; may include idle gaps, not kernel sum')))
    return rows


def region(name):
    return torch.profiler.record_function('AQFC/' + name) if _enabled else nullcontext()


class ProfileLoader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(len(self.loader)):
            with region('DataWait'):
                item = next(iterator)
            yield item


@contextmanager
def training_profile(trace, model, criterion, *, accounting=None, warmup_steps=None,
                     measured_steps=None):
    global _enabled
    if not trace:
        yield
        return
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Profile trace already exists: {path}')
    hooks, stacks = [], {}
    model = model.module if hasattr(model, 'module') else model
    modules = dict(Backbone=model.backbone, Encoder=model.transformer.encoder,
                   AQBA=model.transformer.query_allocator, DGFC=model.transformer.feature_calibrator,
                   DensityPyramid=model.transformer.density_pyramid, Decoder=model.transformer.decoder,
                   Criterion=criterion, Matcher=criterion.matcher)
    def enter(name):
        def callback(module, args):
            scope = region(name); scope.__enter__(); stacks.setdefault(name, []).append(scope)
        return callback
    def leave(name):
        def callback(module, args, output):
            if stacks.get(name):
                stacks[name].pop().__exit__(None, None, None)
        return callback
    _enabled = True
    try:
        for name, module in modules.items():
            if module is not None:
                hooks.append(module.register_forward_pre_hook(enter(name)))
                hooks.append(module.register_forward_hook(leave(name), always_call=True))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            yield
            # Capture before profiler.__exit__ (Kineto postprocessing can be slow).
            postprocess_start = time.perf_counter()
        export_start = time.perf_counter()
        profiler.export_chrome_trace(str(path))
        export_end = time.perf_counter()
        rows = range_summary(profiler.key_averages())
        timing = dict(profiler_finalize_seconds=export_start-postprocess_start,
                      trace_export_seconds=export_end-export_start,
                      summary_aggregation_seconds=time.perf_counter()-export_end)
        if accounting is not None:
            accounting.update(timing)
        path.with_suffix('.summary.json').write_text(json.dumps(dict(
            schema_version=2, ranges=rows, postprocessing=timing,
            scope='entire profiled context, INCLUDING warmup; not measured steps only',
            warmup_steps=warmup_steps, measured_steps=measured_steps,
            warning='CPU/GPU annotation spans and nested ranges must not be summed; not unprofiled throughput'),
            indent=2), encoding='utf-8')
    finally:
        _enabled = False
        for hook in hooks:
            hook.remove()
