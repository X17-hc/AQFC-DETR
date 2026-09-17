"""CPU-only tests: scheduling, provenance, honest failures and paired summaries."""
import json
from pathlib import Path
import pytest

from tools.benchmark_control import (schedule, summarize, parse_gpu_csv,
                                     select_gpu, unique_dir, run_logged)


def test_balanced_order_and_worker_override():
    assert schedule('p1,p2,p2,p1,p2,p1,p1,p2', 0) == [
        ('p1',0),('p2',0),('p2',0),('p1',0),
        ('p2',0),('p1',0),('p1',0),('p2',0)]
    assert schedule('p1,p2',2)==[('p1',2),('p2',2)]
    with pytest.raises(ValueError): schedule('p1,bad',0)
    with pytest.raises(ValueError): schedule('p1,p2',-1)


def test_pairs_and_drift_do_not_select_best_run():
    rows=[dict(variant=v,images_per_second=s) for v,s in
          [('p1',3),('p2',2.94),('p2',2.94),('p1',3)]]
    report=summarize(rows)
    assert report['p2_throughput_penalty_percent']==pytest.approx(2)
    assert report['paired_penalty_percent']==pytest.approx([2,2])
    assert report['throughput_drift_warning'] is False
    rows[-1]['images_per_second']=2.4
    assert summarize(rows)['throughput_drift_warning'] is True
    assert summarize(rows[:1])['p2_throughput_penalty_percent'] is None


def test_missing_telemetry_not_zero_and_device_mapping():
    raw='3, GPU-abc, 00000000:CA:00.0, NVIDIA RTX A6000, 580.82, N/A, 200, 1800, 64, 22500\n'
    rows=parse_gpu_csv(raw)
    assert rows[0]['utilization_percent'] is None
    assert select_gpu(rows,'3')['uuid']=='GPU-abc'
    assert select_gpu(rows,'GPU-abc')['index']=='3'
    with pytest.raises(ValueError): select_gpu(rows,'')
    with pytest.raises(ValueError): select_gpu(rows,'2,3')


def test_output_unique_and_failed_stdout_survives(tmp_path):
    import sys
    a=unique_dir(tmp_path); b=unique_dir(tmp_path)
    assert a!=b
    log=a/'console.log'
    result=run_logged([sys.executable,'-u','-c',
                       'import sys; print("retained"); print("error",file=sys.stderr); sys.exit(7)'],
                      log,a/'telemetry.jsonl',sampler=lambda:{'gpu_error':'unavailable'},interval=.01)
    assert result['returncode']==7
    assert 'retained' in log.read_text() and 'error' in log.read_text()
    assert json.loads((a/'telemetry.jsonl').read_text().splitlines()[0])['gpu_error']=='unavailable'
    with pytest.raises(FileExistsError):
        run_logged([sys.executable,'-c','pass'],log,a/'other.jsonl')


def test_cli_defaults_do_not_import_torch_or_start_experiment():
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    code='from tools.benchmark_interleaved import parser; import sys; assert "torch" not in sys.modules; p=parser(); a=p.parse_args(["--data-root","d","--pretrained","w","--output-dir","o"]); assert a.workers==0 and a.warmup==50 and a.steps==200; assert a.order=="p1,p2,p2,p1,p2,p1,p1,p2"'
    assert subprocess.run([sys.executable,'-c',code],cwd=root).returncode==0


def test_new_run_xml_has_remote_linux_workdir_and_no_console():
    import xml.etree.ElementTree as ET,shlex
    from tools.benchmark_interleaved import parser
    path=Path(__file__).resolve().parents[1]/'.run/AQFC-DETR_InterleavedGPU.run.xml'
    doc=ET.parse(path); options={x.get('name'):x.get('value') for x in doc.findall('.//option')}
    assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
    assert options['RUN_TOOL']==''
    assert doc.find('.//env[@name="CUDA_VISIBLE_DEVICES"]').get('value')=='3'
    args=parser().parse_args(shlex.split(options['PARAMETERS']))
    assert args.workers==0 and args.amp


def test_cpu_deltas_and_telemetry_errors_are_visible(tmp_path):
    import sys
    count=[0]
    def probe():
        count[0]+=1
        return {'cpu_ticks':[20*count[0],0,10*count[0],60*count[0],10*count[0],0,0,0]}
    run_logged([sys.executable,'-c','import time; time.sleep(.15)'],tmp_path/'log',tmp_path/'t',probe,.01)
    samples=[json.loads(x) for x in (tmp_path/'t').read_text().splitlines()]
    assert samples[0]['cpu_busy_percent'] is None
    assert samples[-1]['cpu_busy_percent']==30
    assert samples[-1]['cpu_iowait_percent']==10


def test_telemetry_measured_window_excludes_warmup(tmp_path):
    from tools.benchmark_interleaved import telemetry_review
    path=tmp_path/'t'
    path.write_text('\n'.join(json.dumps(dict(at=f'1970-01-01T00:00:0{i}+00:00',gpus=[
        dict(uuid='GPU-a',sm_clock_mhz=100 if i==0 else 1800)])) for i in range(3)))
    report=telemetry_review(path,{'uuid':'GPU-a'},1,2)
    assert report['available'] and not report['clock_instability_warning']
    assert report['measurement_samples']==2


@pytest.mark.parametrize('fail_at',[None,2])
def test_controller_full_queue_or_first_failure_stops(tmp_path,monkeypatch,fail_at):
    """Drive the real controller with synthetic child results, without CUDA or images."""
    import sys
    import tools.benchmark_interleaved as module
    ann=tmp_path/'data/annotations'; ann.mkdir(parents=True)
    (ann/'aitodv2_trainval.json').write_text(json.dumps({'images':[{'id':i} for i in range(500)]}))
    checkpoint=tmp_path/'weights'; checkpoint.write_bytes(b'fixture-not-a-checkpoint')
    gpu={'index':'3','uuid':'GPU-fixture'}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setattr(module,'snapshot',lambda:{'gpus':[gpu]})
    monkeypatch.setattr(module,'source_hashes',lambda:{'fixture':'unchanged'})
    commands=[]; initializations=[]
    def child(command,log_path,telemetry_path,interval):
        commands.append(command)
        args=module.parser().parse_args(command[3:])
        spec=json.loads(Path(args.worker_spec).read_text())
        initializations.append(spec['checkpoint_sha256'])
        Path(log_path).write_text('fixture child output')
        Path(telemetry_path).write_text('')
        code=9 if fail_at is not None and len(commands)==fail_at else 0
        if code==0:
            result=dict(variant=spec['variant'],images_per_second=3 if spec['variant']=='p1' else 2.94,
                        successful_updates=200,amp_skips=0,
                        measurements=[{'image_ids':spec['image_ids'][i:i+2]} for i in range(100,500,2)],
                        resolved_config={'classification_loss_type':'quality_blend' if spec['variant']=='p2' else 'focal'},
                        measured_start_unix=1,measured_end_unix=2,gpu_identity_verified=True)
            (Path(log_path).parent/'result.json').write_text(json.dumps(result))
        return dict(returncode=code,interrupted=False)
    monkeypatch.setattr(module,'run_logged',child)
    monkeypatch.setattr(module,'telemetry_review',lambda *a:dict(available=True,clock_instability_warning=False))
    monkeypatch.setattr(sys,'argv',['benchmark_interleaved.py','--data-root',str(ann.parent),
        '--pretrained',str(checkpoint),'--output-dir',str(tmp_path/'output')])
    if fail_at is None: module.main()
    else:
        with pytest.raises(RuntimeError,match='no retry'): module.main()
    report=json.loads(next((tmp_path/'output').glob('*/benchmark.json')).read_text())
    assert len(commands)==(8 if fail_at is None else fail_at)
    assert len(set(initializations))==1
    assert all('--resume' not in command for command in commands)
    assert report['status']==('completed' if fail_at is None else 'failed')
    assert not list((tmp_path/'output').rglob('*.pth'))


def test_offline_audit_does_not_promote_partial_results(tmp_path):
    from tools.summarize_gpu_repeat import inspect_run
    assert inspect_run(tmp_path)['status']=='awaiting_metrics'
    row=dict(epoch=0,train_train_iterations=7009,train_optimizer_steps=7006,train_amp_skipped_steps=3,
             train_training_seconds=5000,train_training_phase_epoch=11)
    (tmp_path/'metrics.jsonl').write_text(json.dumps(row)+'\n')
    assert inspect_run(tmp_path)['status']=='partial'
    rows=[dict(row,epoch=i,train_training_phase_epoch=11+i) for i in range(3)]
    rows[-1].update(test_bbox_metrics={'AP':.30},test_evaluated_images=14018,test_evaluation_seconds=6000)
    (tmp_path/'metrics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    assert inspect_run(tmp_path)['status']=='partial'
    for i in range(3): (tmp_path/f'checkpoint{i:04d}.pth').write_bytes(b'not-unpickled')
    assert inspect_run(tmp_path)['status']=='summary_complete'


def test_cuda_uuid_fallback_does_not_guess_device(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from tools.benchmark_control import cuda_context_uuid
    def get_device(pointer): pointer._obj.value=3; return 0
    def get_uuid(pointer,device):
        assert device.value==3
        for i in range(16): pointer._obj[i]=i
        return 0
    monkeypatch.setattr(ctypes,'CDLL',lambda _:SimpleNamespace(cuCtxGetDevice=get_device,cuDeviceGetUuid=get_uuid))
    assert cuda_context_uuid()=='GPU-00010203-0405-0607-0809-0a0b0c0d0e0f'
    monkeypatch.setattr(ctypes,'CDLL',lambda _:SimpleNamespace(cuCtxGetDevice=lambda _:201))
    with pytest.raises(RuntimeError,match='201'): cuda_context_uuid()


def test_initial_eval_is_not_incorrectly_required_to_train(tmp_path):
    from tools.summarize_gpu_repeat import inspect_run
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(test_bbox_metrics={'AP':.23},test_evaluated_images=14018)))
    assert inspect_run(tmp_path,eval_only=True)['status']=='evaluation_summary_complete'
    assert inspect_run(tmp_path)['status']=='partial'


def test_timeout_is_advisory_not_a_kill(tmp_path):
    import sys
    row=run_logged([sys.executable,'-c','import time; time.sleep(.05); print("finished")'],
                   tmp_path/'out',tmp_path/'t',sampler=lambda:{},interval=.01,warning_after=.001)
    assert row['returncode']==0 and row['time_budget_warning']
    assert 'finished' in (tmp_path/'out').read_text()
