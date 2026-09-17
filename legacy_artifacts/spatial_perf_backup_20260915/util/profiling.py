"""Opt-in trace ranges; normal training does not activate hooks or CUDA sync."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import torch

_enabled = False


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
def training_profile(trace, model, criterion):
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
        profiler.export_chrome_trace(str(path))
        rows = [dict(name=e.key, cpu_total_us=e.cpu_time_total,
                     device_total_us=e.device_time_total, count=e.count)
                for e in profiler.key_averages() if e.key.startswith('AQFC/')]
        path.with_suffix('.summary.json').write_text(json.dumps(dict(
            ranges=rows, warning='Nested instrumented ranges overlap; not additive or unprofiled throughput'),
            indent=2), encoding='utf-8')
    finally:
        _enabled = False
        for hook in hooks:
            hook.remove()
