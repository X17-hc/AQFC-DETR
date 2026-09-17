"""Standard-library-only control plane; never imports CUDA or edits training state."""
import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    # This directory belongs to this invocation; never used on historical results.
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def unique_dir(root):
    path=Path(root)/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}'
    path.mkdir(parents=True,exist_ok=False)
    return path.resolve()


def schedule(order,workers,variants=('p1','p2')):
    values=[v.strip() for v in order.split(',')]
    if workers not in (0,2,4,8) or not values or any(v not in variants for v in values):
        raise ValueError(f'Order must contain {variants}; workers must be 0,2,4,8')
    return [(v,workers) for v in values]


def summarize(rows):
    speeds={v:[r['images_per_second'] for r in rows if r['variant']==v] for v in ('p1','p2')}
    if any(not math.isfinite(x) or x<=0 for xs in speeds.values() for x in xs):
        raise ValueError('Invalid throughput; retain failure, do not discard the row')
    med={v:statistics.median(xs) if xs else None for v,xs in speeds.items()}
    pairs=[]
    for i in range(0,len(rows)-1,2):
        pair={r['variant']:r['images_per_second'] for r in rows[i:i+2]}
        if set(pair)=={'p1','p2'}: pairs.append(100*(1-pair['p2']/pair['p1']))
    # Advisory predeclared range test; this is not proof of resource contention.
    ranges={v:100*(max(xs)-min(xs))/statistics.median(xs) if len(xs)>1 else None
            for v,xs in speeds.items()}
    return dict(median_images_per_second=med,paired_penalty_percent=pairs,
                p2_throughput_penalty_percent=100*(1-med['p2']/med['p1']) if all(med.values()) else None,
                within_variant_range_percent=ranges,
                throughput_drift_warning=any(x is not None and x>10 for x in ranges.values()))


def summarize_spatial(rows):
    """Both arms use P2; do not mislabel backend speed as a focal/quality comparison."""
    mapped=[dict(r,variant={'reference':'p1','optimized':'p2'}[r['variant']]) for r in rows]
    data=summarize(mapped)
    med=data['median_images_per_second']
    return dict(comparison='P2 reference vs P2 optimized spatial selection',
        median_images_per_second={'reference':med['p1'],'optimized':med['p2']},
        optimized_throughput_gain_percent=(-data['p2_throughput_penalty_percent']
            if data['p2_throughput_penalty_percent'] is not None else None),
        paired_gain_percent=[-x for x in data['paired_penalty_percent']],
        within_variant_range_percent={k:data['within_variant_range_percent'][v]
            for k,v in [('reference','p1'),('optimized','p2')]},
        throughput_drift_warning=data['throughput_drift_warning'])


def parse_gpu_csv(text):
    fields=('index','uuid','pci_bus_id','name','driver','utilization_percent','power_w',
            'sm_clock_mhz','temperature_c','memory_used_mib')
    rows=[]
    for cells in csv.reader(io.StringIO(text)):
        if not cells: continue
        if len(cells)!=len(fields): raise ValueError('Unexpected nvidia-smi CSV columns')
        row=dict(zip(fields,(c.strip() for c in cells)))
        for key in fields[5:]:
            try:
                value=float(row[key]); row[key]=value if math.isfinite(value) else None
            except ValueError: row[key]=None
        rows.append(row)
    return rows


def select_gpu(rows,visible):
    # Selection is owned by the launch environment, not hardcoded to GPU 2 or 3.
    if not visible or ',' in visible: raise ValueError('Set exactly one CUDA_VISIBLE_DEVICES in the run configuration')
    matches=[r for r in rows if r['index']==visible or r['uuid']==visible]
    if len(matches)!=1: raise ValueError('Cannot unambiguously map CUDA_VISIBLE_DEVICES to GPU UUID')
    return matches[0]


def cuda_context_uuid():
    """PyTorch 2.4 fallback: identity of the calling thread's EXISTING CUDA context.

    Called only inside the user-launched CUDA worker, never by the CPU controller.
    Driver status failures are visible; never substitutes the requested index as proof.
    """
    import ctypes
    library=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
    device=ctypes.c_int()
    status=library.cuCtxGetDevice(ctypes.byref(device))
    if status: raise RuntimeError(f'cuCtxGetDevice status {status}')
    raw=(ctypes.c_ubyte*16)()
    query=getattr(library,'cuDeviceGetUuid_v2',None) or library.cuDeviceGetUuid
    status=query(ctypes.byref(raw),device)
    if status: raise RuntimeError(f'cuDeviceGetUuid status {status}')
    return 'GPU-'+str(uuid.UUID(bytes=bytes(raw)))


def snapshot():
    result={'at':utc(),'gpu_process_ownership':'unknown_container_visibility'}
    try:
        args=['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,driver_version,utilization.gpu,power.draw,clocks.sm,temperature.gpu,memory.used','--format=csv,noheader,nounits']
        p=subprocess.run(args,capture_output=True,text=True,timeout=3,check=True)
        result['gpus']=parse_gpu_csv(p.stdout)
    except (OSError,subprocess.SubprocessError,ValueError) as e:
        result.update(gpus=None,gpu_error=str(e))
    try:
        # Host aggregate counters only: deltas are computed offline. No new dependency.
        result['cpu_ticks']=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
        result['load_average']=list(os.getloadavg())
        result['cpu_scope']='visible_system_aggregate_not_training_process'
    except (OSError,AttributeError,ValueError):
        result.update(cpu_ticks=None,load_average=None,cpu_scope='unavailable')
    return result


def run_logged(command,log_path,telemetry_path,sampler=snapshot,interval=5,warning_after=1800):
    """One child only. Failure retained; never retries, kills, or changes GPU choice."""
    if interval<=0: raise ValueError('Telemetry interval must be positive')
    start=utc(); begin=time.monotonic()
    with Path(log_path).open('x',encoding='utf-8') as log, Path(telemetry_path).open('x',encoding='utf-8') as telemetry:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        interrupted=False; previous_cpu=None
        while True:
            try:
                sample=sampler()
            except Exception as e: sample={'at':utc(),'telemetry_error':repr(e)}
            sample.update(child_pid=proc.pid,child_alive=proc.poll() is None,
                          elapsed_seconds=time.monotonic()-begin,
                          time_budget_warning=time.monotonic()-begin>warning_after)
            ticks=sample.get('cpu_ticks')
            sample.update(cpu_busy_percent=None,cpu_iowait_percent=None)
            if ticks and previous_cpu:
                # Linux guest ticks are already included in user/nice: use first 8 only.
                delta=[b-a for a,b in zip(previous_cpu[:8],ticks[:8])]
                total=sum(delta)
                if len(delta)>=5 and total>0 and min(delta)>=0:
                    sample['cpu_busy_percent']=100*(total-delta[3]-delta[4])/total
                    sample['cpu_iowait_percent']=100*delta[4]/total
            if ticks: previous_cpu=ticks
            telemetry.write(json.dumps(sample,allow_nan=False)+'\n'); telemetry.flush()
            try:
                code=proc.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired: pass
            except KeyboardInterrupt:
                # Do not kill a running group; stop scheduling after it exits.
                interrupted=True
        return dict(returncode=code,pid=proc.pid,start_utc=start,end_utc=utc(),
                    end_to_end_seconds=time.monotonic()-begin,interrupted=interrupted,
                    time_budget_warning=time.monotonic()-begin>warning_after)
