"""Regress the actual loop's timing placement and profiler summary contract."""
import ast
from pathlib import Path
import importlib.util
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def test_measurement_end_is_captured_inside_profile_before_export():
    tree=ast.parse((ROOT/'tools/benchmark_incremental.py').read_text(encoding='utf-8'))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    scope=next(n for n in run.body if isinstance(n,ast.With))
    # Placement guard on the real loop, then execute its endpoint assignment
    # against a fake clock that advances by 180 seconds after leaving the loop.
    assignments=[n for n in ast.walk(scope) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='measured_end_unix' for t in n.targets)]
    assert assignments, 'measurement endpoint is currently recorded AFTER profiler export'
    assert not any(isinstance(n,ast.keyword) and n.arg=='measured_end_unix'
                   and isinstance(n.value,ast.Call) for n in ast.walk(run))
    clock=[100.]
    namespace={'time':SimpleNamespace(time=lambda:clock[0])}
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[])), '<endpoint>', 'exec'),namespace)
    clock[0]+=180
    assert namespace['measured_end_unix']==100.


def test_summary_keeps_cpu_and_gpu_annotations_distinct():
    spec=importlib.util.spec_from_file_location('profile_under_test',ROOT/'util/profiling.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    events=[SimpleNamespace(key='AQFC/Encoder',device_type=d,cpu_time_total=c,
        device_time_total=g,self_cpu_time_total=c/2,self_device_time_total=g/2,count=21)
        for d,c,g in [('DeviceType.CPU',100.,50.),('DeviceType.CUDA',0.,80.)]]
    rows=module.range_summary(events)
    assert [r['device_type'] for r in rows]==['CPU','CUDA']
    assert rows[0]['timing_semantics'] != rows[1]['timing_semantics']
    assert rows[0]['self_cpu_us']==50.
