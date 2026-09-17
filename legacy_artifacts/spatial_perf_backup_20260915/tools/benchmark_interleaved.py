"""MANUAL ONLY: fresh-process P1/P2 interleaving, reusing benchmark_incremental.run.

No training checkpoints are saved. GPU selection is exclusively an environment setting.
Default schedule: ABBA BAAB, workers=0. Existing benchmark entry remains unchanged.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.benchmark_control import (digest,save,unique_dir,schedule,summarize,
                                     snapshot,select_gpu,run_logged,utc,cuda_context_uuid)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--pretrained',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--order',default='p1,p2,p2,p1,p2,p1,p1,p2')
    p.add_argument('--workers',type=int,choices=[0,2,4,8],default=0)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--profile',action='store_true',help='Separate manual profile; never a speed conclusion')
    p.add_argument('--telemetry-seconds',type=float,default=5)
    p.add_argument('--worker-spec',help=argparse.SUPPRESS)
    return p


def source_hashes():
    paths={ROOT/'main.py',ROOT/'engine.py'}
    for folder in ('models','datasets','util','configs','tools'):
        paths.update((ROOT/folder).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}


def worker(cli):
    """One clean interpreter = fresh RNG/model/optimizer/CUDA allocator for each repeat."""
    spec=json.loads(Path(cli.worker_spec).read_text(encoding='utf-8'))
    if source_hashes()!=spec['source_sha256']: raise RuntimeError('Source changed during benchmark queue')
    if digest(cli.pretrained)!=spec['checkpoint_sha256']: raise RuntimeError('Initialization changed')
    annotation=Path(cli.data_root)/'annotations/aitodv2_trainval.json'
    if digest(annotation)!=spec['annotation_sha256']: raise RuntimeError('Annotation changed')
    import torch
    from tools.benchmark_incremental import run
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    props=torch.cuda.get_device_properties(0)
    actual_uuid=getattr(props,'uuid',None)
    actual_uuid=str(actual_uuid) if actual_uuid is not None else None
    uuid_error=None
    if actual_uuid is None:
        try:
            # Materialize this worker's context; zero elements, outside measured time.
            torch.empty(0,device='cuda')
            actual_uuid=cuda_context_uuid()
        except (OSError,AttributeError,RuntimeError) as exc: uuid_error=repr(exc)
    if actual_uuid is not None and actual_uuid!=spec['gpu']['uuid']:
        raise RuntimeError(f'CUDA/nvidia-smi UUID mismatch: {actual_uuid}')
    cli.require_square_800=True
    row=run(spec['variant'],cli.workers,spec['sequence_index'],cli,spec['image_ids'])
    row.update(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
               cuda_logical_device=0,cuda_uuid=actual_uuid,
               cuda_uuid_error=uuid_error,
               gpu_identity_verified=actual_uuid==spec['gpu']['uuid'],
               checkpoint_sha256=spec['checkpoint_sha256'],
               image_ids_sha256=spec['image_ids_sha256'])
    save(Path(cli.output_dir)/'result.json',row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('measurements','resolved_config')}),flush=True)


def telemetry_review(path,gpu,start,end):
    from datetime import datetime
    samples=[json.loads(line) for line in Path(path).read_text().splitlines()]
    measured=[]; errors=[]
    for s in samples:
        if s.get('gpu_error') or s.get('telemetry_error'): errors.append(s)
        if not s.get('at'): continue
        timestamp=datetime.fromisoformat(s['at']).timestamp()
        if start<=timestamp<=end:
            matched=[g for g in (s.get('gpus') or []) if g['uuid']==gpu['uuid']]
            if matched: measured.append(matched[0])
    clocks=[s['sm_clock_mhz'] for s in measured if s['sm_clock_mhz'] is not None]
    import statistics
    ratio=(max(clocks)-min(clocks))/statistics.median(clocks) if clocks and statistics.median(clocks)>0 else None
    return dict(measurement_samples=len(measured),errors=len(errors),
                available=len(measured)>=2 and not errors,
                clock_range_percent=ratio*100 if ratio is not None else None,
                clock_instability_warning=ratio is not None and ratio>.15,
                ownership='unknown_even_if_process_table_is_empty')


def main():
    p=parser(); cli=p.parse_args()
    if min(cli.warmup,cli.steps,cli.telemetry_seconds)<=0: p.error('Counts/interval must be positive')
    if cli.worker_spec:
        worker(cli); return
    jobs=schedule(cli.order,cli.workers)
    if cli.profile:
        if cli.order!='p1,p2': p.error('Profile requires explicit --order p1,p2')
        cli.warmup=1; cli.steps=20
    root=unique_dir(cli.output_dir)
    record=dict(version='gpu_interleaved_v1',status='preparing',created_at=utc(),
                argv=sys.argv,results=[],runs=[],schedule=jobs,purpose='profile' if cli.profile else 'benchmark',
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',
                    'OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONIOENCODING')},
                warmup=cli.warmup,steps=cli.steps,amp=cli.amp,
                limitations=['Resource ownership unknown in container','No statistical significance test',
                             'Transient training updates; no checkpoint saved'])
    print(f'Output: {root}',flush=True)
    with (root/'controller.log').open('x',encoding='utf-8') as log:
        def announce(message):
            print(message,flush=True); log.write(message+'\n'); log.flush()
        try:
            first=snapshot(); save(root/'resources_before.json',first)
            gpu=select_gpu(first.get('gpus') or [],record['environment']['CUDA_VISIBLE_DEVICES'])
            ann=Path(cli.data_root)/'annotations/aitodv2_trainval.json'
            raw=json.loads(ann.read_text(encoding='utf-8'))
            ids=sorted(x['id'] for x in raw['images'])
            if len(ids)!=len(set(ids)): raise ValueError('Duplicate annotation image IDs')
            random.Random(42).shuffle(ids); ids=ids[:2*(cli.warmup+cli.steps)]
            if len(ids)!=2*(cli.warmup+cli.steps): raise ValueError('Not enough images')
            save(root/'image_ids.json',ids)
            record.update(source_sha256=source_hashes(),checkpoint_sha256=digest(cli.pretrained),
                          annotation_sha256=digest(ann),image_ids=ids,
                          image_ids_sha256=digest(root/'image_ids.json'),gpu=gpu,status='running')
            save(root/'benchmark.json',record)
            for index,(variant,workers) in enumerate(jobs):
                if source_hashes()!=record['source_sha256']: raise RuntimeError('Source changed; queue stopped')
                directory=root/f'{index:02d}_{variant}_w{workers}'; directory.mkdir()
                spec={k:record[k] for k in ('source_sha256','checkpoint_sha256','annotation_sha256',
                                          'image_ids','image_ids_sha256','gpu')}
                spec.update(variant=variant,sequence_index=index)
                save(directory/'spec.json',spec)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--data-root',cli.data_root,
                         '--pretrained',cli.pretrained,'--output-dir',str(directory),
                         '--workers',str(workers),'--warmup',str(cli.warmup),'--steps',str(cli.steps),
                         '--amp' if cli.amp else '--no-amp','--worker-spec',str(directory/'spec.json')]
                if cli.profile: command.append('--profile')
                announce(f'[{index+1}/{len(jobs)}] {variant}, workers={workers}; full log: {directory}/console.log')
                entry=dict(command=command,variant=variant,sequence_index=index,directory=str(directory))
                record['runs'].append(entry); save(root/'benchmark.json',record)
                entry.update(run_logged(command,directory/'console.log',directory/'telemetry.jsonl',interval=cli.telemetry_seconds))
                if entry['returncode']!=0: raise RuntimeError(f'{variant} exit {entry["returncode"]}; no retry')
                row=json.loads((directory/'result.json').read_text(encoding='utf-8'))
                if row['successful_updates']<=0: raise RuntimeError('No successful measured optimizer update')
                if len(row['measurements'])!=cli.steps: raise RuntimeError('Incomplete measurement coverage')
                expected_ids=ids[2*cli.warmup:]
                actual_ids=[i for item in row['measurements'] for i in item['image_ids']]
                if actual_ids!=expected_ids: raise RuntimeError('Measured image order changed')
                if record['results']:
                    reference=record['results'][0]['resolved_config']
                    changed=[k for k in set(reference)|set(row['resolved_config'])
                             if reference.get(k)!=row['resolved_config'].get(k) and k!='classification_loss_type']
                    if changed: raise RuntimeError(f'Unexpected config differences: {changed}')
                row['telemetry']=telemetry_review(directory/'telemetry.jsonl',gpu,row['measured_start_unix'],row['measured_end_unix'])
                record['results'].append(row); record['summary']=summarize(record['results'])
                announce(json.dumps(dict(variant=variant,images_per_second=row['images_per_second'],
                                         amp_skips=row['amp_skips'],telemetry=row['telemetry'])))
                save(root/'benchmark.json',record)
                # No occupancy guard; completed-measurement validity decides whether to schedule MORE work.
                if (entry['interrupted'] or entry.get('time_budget_warning',False) or not row['gpu_identity_verified'] or not row['telemetry']['available'] or
                    row['telemetry']['clock_instability_warning'] or record['summary']['throughput_drift_warning']):
                    record['status']='paused_after_group'; record['reason']='Interrupt, group over 30 minutes, missing identity/telemetry or predeclared drift warning; inspect retained data before further work'
                    break
            else: record['status']='completed'
            if record['status']=='completed' and not cli.profile:
                record['next_action']='Review telemetry and paired penalties; workers2/profile require a separate manual launch, never automatic'
        except BaseException as exc:
            record.update(status='failed',error=repr(exc)); log.write(traceback.format_exc()); raise
        finally:
            record['ended_at']=utc(); save(root/'benchmark.json',record)
            announce(f'Status: {record["status"]}; no formal checkpoint modified')


if __name__=='__main__': main()
