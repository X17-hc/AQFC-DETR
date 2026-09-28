"""MANUAL accepted-student ABBA training benchmark, no checkpoint writes."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import uuid
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.benchmark_incremental import run
from util.experiment import sha256
from util.p2_transfer import SOURCE_SHA
from tools.benchmark_control import snapshot,select_gpu,cuda_context_uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','pretrained','output-dir','adapter','acceptance'): p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--warmup',type=int,default=50); p.add_argument('--steps',type=int,default=200)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    cli=p.parse_args()
    if min(cli.warmup,cli.steps)<1: raise ValueError('Positive warmup and measurement steps required')
    receipt=json.loads(Path(cli.acceptance).read_text())
    if (receipt.get('accepted') is not True or receipt.get('adapter_sha256')!=sha256(cli.adapter)
            or receipt.get('image_count')!=1000 or sha256(cli.pretrained)!=SOURCE_SHA):
        raise ValueError('Matched accepted adapter and original S2 source are required')
    cli.profile=False; cli.require_square_800=True
    out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False); cli.output_dir=str(out)
    annotation=Path(cli.data_root)/'annotations/aitodv2_trainval.json'
    ids=sorted(x['id'] for x in json.loads(annotation.read_text())['images'])
    random.Random(42).shuffle(ids); ids=ids[:2*(cli.warmup+cli.steps)]
    if len(ids)!=2*(cli.warmup+cli.steps): raise ValueError('Insufficient fixed images')
    report=dict(command=sys.argv,image_ids=ids,source_sha256=SOURCE_SHA,adapter_sha256=sha256(cli.adapter),
        annotation_sha256=sha256(annotation),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),results=[],
        limitation='No profile; no teacher; transient detector updates; resource ownership not established')
    resources=snapshot()
    report['gpu']=select_gpu(resources.get('gpus') or [],os.environ.get('CUDA_VISIBLE_DEVICES'))
    torch.empty(0,device='cuda')
    actual_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid',None) or cuda_context_uuid())
    if actual_uuid!=report['gpu']['uuid']: raise RuntimeError('CUDA device UUID differs from selected GPU')
    report['environment']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cuda_uuid=actual_uuid)
    (out/'resources_before.json').write_text(json.dumps(resources,indent=2))
    print(f'Output: {out}',flush=True)
    from tools.train_p2_transfer import Tee
    for index,variant in enumerate(('f0','p2_student','p2_student','f0')):
        stop=threading.Event()
        def sample_resources():
            with (out/f'{index}_{variant}_telemetry.jsonl').open('x') as stream:
                while not stop.is_set():
                    try: item=snapshot()
                    except Exception as exc: item=dict(at_unix=time.time(),error=repr(exc))
                    stream.write(json.dumps(item)+'\n'); stream.flush()
                    stop.wait(5)
        sampler=threading.Thread(target=sample_resources,daemon=True); sampler.start()
        with (out/f'{index}_{variant}.log').open('x') as log:
            stdout,stderr=sys.stdout,sys.stderr; sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
            try: row=run(variant,cli.workers,index,cli,ids)
            finally:
                sys.stdout,sys.stderr=stdout,stderr
                stop.set(); sampler.join(timeout=30)
                if sampler.is_alive(): raise RuntimeError('Telemetry did not stop; queue halted')
        report['results'].append(row)
        (out/'benchmark.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(variant=variant,images_per_second=row['images_per_second'],amp_skips=row['amp_skips'])),flush=True)
    base=statistics.median(r['images_per_second'] for r in report['results'] if r['variant']=='f0')
    student=statistics.median(r['images_per_second'] for r in report['results'] if r['variant']=='p2_student')
    report.update(throughput_gain_percent=100*(student/base-1),initial_five_percent_gate=student>=base*1.05)
    (out/'benchmark.json').write_text(json.dumps(report,indent=2))
    print(f"Student throughput gain: {report['throughput_gain_percent']:.2f}% (resource comparability must be reviewed)")


if __name__=='__main__': main()
