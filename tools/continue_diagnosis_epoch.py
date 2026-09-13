"""Approved bounded continuation: offline -> floor check -> one resumed epoch -> 640 eval.

No retries or configuration search. Uses immutable prior exports and a new output root.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_factor_diagnosis import DATA, UPDATED, EXPECTED, read, save, source_hashes, artifacts
from util.experiment import sha256
OLD = ROOT/'outputs/factor_diagnosis_20260909_084529'


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    assert sha256(UPDATED) == EXPECTED
    for name in ('subset_ids.json','precheck_ids.json','updated_config.py'):
        shutil.copy2(OLD/name, output/name)
    shutil.copy2(UPDATED.parent/'config_cfg.py', output/'training_config.py')
    annotation = DATA/'annotations/aitodv2_test.json'
    manifest = dict(started_at='2026-09-10T01:25:34+08:00',
        started_epoch=datetime.fromisoformat('2026-09-10T01:25:34+08:00').timestamp(),
        source=source_hashes(), checkpoint=str(UPDATED), checkpoint_sha256=EXPECTED,
        immutable={str(p):sha256(p) for p in [annotation,DATA/'annotations/aitodv2_trainval.json',
            output/'subset_ids.json',output/'precheck_ids.json',output/'updated_config.py',output/'training_config.py']},
        previous_results=str(OLD), purpose='engineering_check; trainval/test; not independent research validation')
    save(output/'manifest.json',manifest)
    save(output/'state.json',dict(status='prepared',runs=[]))
    print(str(output),flush=True)


def predictions(directory):
    path = next(directory.glob('predictions_*/predictions.json'))
    return read(path)


def offline(output):
    from util.error_analysis import analyze
    annotation = read(DATA/'annotations/aitodv2_test.json')
    ids = read(output/'subset_ids.json')
    root = output/'offline'
    root.mkdir(exist_ok=False)
    diag_path = next((OLD/'S1').glob('predictions_*/images.jsonl'))
    diagnostics = {r['image_id']:r for r in map(json.loads,diag_path.read_text().splitlines())}
    rng = random.Random(42)
    selected = []
    for level in (300,500,900,1500):
        group = sorted(i for i,r in diagnostics.items() if r['raw_query_counts']==level)
        selected.extend(sorted(rng.sample(group,min(6,len(group)))))
    save(root/'visualization_ids.json',selected)
    summaries, selected_rows = {}, {}
    for name in ('S1','S2','S3','S4'):
        result = analyze(predictions(OLD/name), annotation, ids, confidence=.25, iou_threshold=.5)
        compact = []
        selected_rows[name] = {}
        for row in result.pop('images'):
            counts = Counter(row['prediction_status'])
            compact.append(dict(image_id=row['image_id'],gt=len(row['ground_truth']),
                tp=counts['tp'],fp=counts['fp']+counts['duplicate'],duplicate=counts['duplicate'],
                fn=len(row['missed_gt']),original_K=diagnostics[row['image_id']]['raw_query_counts']))
            if row['image_id'] in selected:
                selected_rows[name][row['image_id']] = row
        result['images'] = compact
        result['by_original_K'] = {str(k):{key:sum(r[key] for r in compact if r['original_K']==k)
            for key in ('gt','tp','fp','fn','duplicate')} for k in (300,500,900,1500)}
        save(root/f'{name}.json',result)
        summaries[name] = {k:v for k,v in result.items() if k!='images'}
        print(f'Offline {name} complete',flush=True)
    save(root/'summary.json',summaries)
    # Existing local image annotation only; no AI-generated evidence, no package installation.
    from PIL import Image, ImageDraw
    files = {x['id']:x['file_name'] for x in annotation['images']}
    for image_id in selected:
        original = Image.open(DATA/'images/test/images'/files[image_id]).convert('RGB')
        panel = Image.new('RGB',(1600,1660),'white')
        for j,name in enumerate(('S1','S2','S3','S4')):
            canvas = original.copy()
            draw = ImageDraw.Draw(canvas)
            row = selected_rows[name][image_id]
            for pred,status in zip(row['predictions'],row['prediction_status']):
                x,y,w,h = pred['bbox']
                draw.rectangle((x,y,x+w,y+h),outline={'tp':'lime','fp':'yellow','duplicate':'magenta'}[status],width=1)
            for index in row['missed_gt']:
                x,y,w,h=row['ground_truth'][index]['bbox']
                draw.rectangle((x,y,x+w,y+h),outline='red',width=1)
            col,row_index=j%2,j//2
            panel.paste(canvas.resize((800,800)),(col*800,row_index*830+30))
            ImageDraw.Draw(panel).text((col*800+10,row_index*830+8),f'{name} image={image_id}: TP green / FP yellow / duplicate magenta / FN red',fill='black')
        panel.save(root/f'comparison_{image_id}.jpg',quality=90)


def cmd(output,name,checkpoint,ids=None,floor=0,train=False):
    command=[sys.executable,'-u',str(ROOT/'main.py'),'--config',
        str(output/('training_config.py' if train else 'updated_config.py')),
        '--data-root',str(DATA),'--no-pretrained','--resume',str(checkpoint),
        '--output-dir',str(output/name),'--device','cuda','--amp','--seed','42','--num_workers','0',
        '--max-train-steps','0','--max-eval-steps','0']
    if train:
        command += ['--max-consecutive-skipped-steps','20','--options','epochs=2','val_epoch=[]']
    else:
        command += ['--eval','--export-predictions','--export-diagnostics','--eval-query-floor',str(floor)]
        if ids:
            command += ['--eval-image-ids',str(output/ids)]
        command += ['--options','data_aug_scales=[640]','data_aug_max_size=640']
    return command


def compare_predictions(before,after):
    import numpy as np
    assert len(before)==len(after)
    for a,b in zip(before,after):
        assert (a['image_id'],a['category_id'])==(b['image_id'],b['category_id'])
        np.testing.assert_allclose([*a['bbox'],a['score']],[*b['bbox'],b['score']],atol=1e-5,rtol=1e-4)


def run(output):
    with (output/'RUN_STARTED').open('x') as f:
        f.write(datetime.now().astimezone().isoformat())
    manifest=read(output/'manifest.json')
    state=read(output/'state.json')
    state['queue_pid']=os.getpid()
    def integrity():
        if source_hashes()!=manifest['source']:
            raise ValueError('Source modified after preparation')
        if sha256(UPDATED)!=EXPECTED:
            raise ValueError('Original checkpoint modified')
        for p,digest in manifest['immutable'].items():
            if sha256(p)!=digest:
                raise ValueError(f'Immutable input changed: {p}')
        if shutil.disk_usage(output).free < 6*1024**3:
            raise OSError('Less than 6 GiB disk free')
    def execute(name,checkpoint=UPDATED,ids='precheck_ids.json',floor=0,train=False):
        integrity()
        if time.time()-manifest['started_epoch']>=18*3600:
            raise TimeoutError('18-hour new-start cutoff reached')
        path=output/name
        path.mkdir(exist_ok=False)
        command=cmd(output,name,checkpoint,ids,floor,train)
        record=dict(name=name,command=command,start=time.time(),checkpoint_sha256=sha256(checkpoint),status='running')
        state['runs'].append(record)
        state.update(status='running',current=name)
        save(output/'state.json',state)
        with (path/'console.log').open('w',encoding='utf-8') as out, (path/'stderr.log').open('w',encoding='utf-8') as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1'))
            record['pid']=process.pid
            save(output/'state.json',state)
            record['exit_code']=process.wait()
        record['seconds']=time.time()-record['start']
        if record['exit_code']:
            raise RuntimeError(f'{name} failed; no retry')
        if train:
            run_manifest=read(path/'experiment_manifest.json')
            stats=json.loads((path/'log.txt').read_text().splitlines()[-1])
            if run_manifest.get('last_epoch')!=1 or not run_manifest.get('epoch_complete'):
                raise ValueError('Additional epoch incomplete')
            if stats['train_optimizer_steps']<=0 or not (path/'checkpoint.pth').is_file():
                raise ValueError('No completed training checkpoint/update')
            record['train_stats']=stats
        else:
            expected=read(output/ids) if ids else [x['id'] for x in read(DATA/'annotations/aitodv2_test.json')['images']]
            stats,_=artifacts(path,expected,True)
            record.update(metrics=stats['test_bbox_metrics'],images=stats['test_evaluated_images'],queries=stats['test_decoder_query_tokens'])
        record['status']='completed'
        save(output/'state.json',state)
        print(f'{name} completed in {record["seconds"]:.1f}s',flush=True)
        return record
    try:
        integrity()
        state.update(status='running',current='offline')
        save(output/'state.json',state)
        offline(output)
        execute('precheck_default')
        compare_predictions(predictions(OLD/'precheck_on'),predictions(output/'precheck_default'))
        execute('precheck_floor',floor=900)
        baseline_file=next((output/'precheck_default').glob('predictions_*/images.jsonl'))
        rows=[json.loads(x) for x in baseline_file.read_text().splitlines()]
        high={r['image_id'] for r in rows if r['executed_query_count']>=900}
        compare_predictions([p for p in predictions(output/'precheck_default') if p['image_id'] in high],
                            [p for p in predictions(output/'precheck_floor') if p['image_id'] in high])
        floor_file=next((output/'precheck_floor').glob('predictions_*/images.jsonl'))
        floor_rows={r['image_id']:r for r in map(json.loads,floor_file.read_text().splitlines())}
        for row in rows:
            assert floor_rows[row['image_id']]['executed_query_count']==max(900,row['executed_query_count'])
        save(output/'precheck_passed.json',dict(passed=True,images=32,high_budget_images=len(high),atol=1e-5,rtol=1e-4))
        execute('B1_floor900',ids='subset_ids.json',floor=900)
        # Local training checkpoint only: same deserialization as the existing trusted resume loader.
        import torch
        checkpoint=torch.load(UPDATED,map_location='cpu',weights_only=False)
        required=('model','optimizer','lr_scheduler','scaler','rng_states','epoch','variant_signature','best_metrics')
        if any(k not in checkpoint for k in required) or checkpoint['epoch']!=0 or not checkpoint['rng_states']:
            raise ValueError('Original checkpoint cannot satisfy strict epoch-1 resume')
        save(output/'resume_preflight.json',dict(required_keys_present=True,source_epoch=0,source_sha256=EXPECTED))
        del checkpoint
        execute('train_epoch1',ids=None,train=True)
        new=output/'train_epoch1/checkpoint.pth'
        subset=execute('epoch1_subset640',checkpoint=new,ids='subset_ids.json')
        estimate=subset['seconds']/1000*14018*1.5+600
        if time.time()+estimate > manifest['started_epoch']+22*3600:
            state.update(status='completed_partial',note='Full evaluation deferred by 22-hour completion budget')
        else:
            execute('epoch1_full640',checkpoint=new,ids=None)
            state['status']='completed'
        save(output/'state.json',state)
    except Exception as exc:
        state.update(status='paused_error',error=f'{type(exc).__name__}: {exc}')
        save(output/'state.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    (prepare if args.prepare else run)(args.output.resolve())
