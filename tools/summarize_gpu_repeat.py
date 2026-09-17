"""Offline JSON-only audit. Never runs inference or unpickles checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.benchmark_control import unique_dir,save,digest,utc


def inspect_run(path,eval_only=False):
    path=Path(path); metrics=path/'metrics.jsonl'
    result=dict(path=str(path),status='awaiting_metrics',training=[],evaluation=None,files={})
    for name in ('metrics.jsonl','experiment_manifest.json','config_args_all.json'):
        source=path/name
        if source.exists(): result['files'][name]=dict(sha256=digest(source),bytes=source.stat().st_size)
    if not metrics.exists(): return result
    rows=[json.loads(x) for x in metrics.read_text(encoding='utf-8').splitlines() if x.strip()]
    for r in rows:
        if 'train_train_iterations' in r:
            keys=('training_seconds','train_iterations','optimizer_steps','amp_skipped_steps',
                  'training_phase_epoch','mean_executed_queries','quality_lambda')
            result['training'].append(dict(epoch=r['epoch'],**{k:r.get('train_'+k) for k in keys}))
    for r in rows:
        if 'test_bbox_metrics' in r:
            result['evaluation']=dict(metrics=r['test_bbox_metrics'],evaluated_images=r.get('test_evaluated_images'),
                seconds=r.get('test_evaluation_seconds'),query_tokens=r.get('test_decoder_query_tokens'))
    train=result['training']
    result['training_complete']=(len(train)==3 and [r['epoch'] for r in train]==[0,1,2] and
        all(r['train_iterations']==7009 and r['optimizer_steps'] is not None and r['optimizer_steps']>0 and
            r['amp_skipped_steps'] is not None and r['optimizer_steps']+r['amp_skipped_steps']==7009 and
            r['training_phase_epoch']==11+r['epoch'] for r in train))
    result['checkpoints_present']={f'checkpoint{i:04d}.pth':(path/f'checkpoint{i:04d}.pth').is_file() for i in range(3)}
    result['training_seconds']=sum(r['training_seconds'] for r in train if r['training_seconds'] is not None)
    result['evaluation_count_complete']=bool(result['evaluation'] and result['evaluation']['evaluated_images']==14018)
    result['status']='summary_complete' if result['training_complete'] and result['evaluation_count_complete'] and all(result['checkpoints_present'].values()) else 'partial'
    if eval_only and result['evaluation_count_complete']:
        result['status']='evaluation_summary_complete'
    result['limitations']=['Image ID equality and checkpoint internal state require separate validation; count and presence are not proof',
                           'Process exit is not inferred from a metrics file']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('r0','p1','old-p2','new-p2'): p.add_argument('--'+name,required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args(); root=unique_dir(args.output_dir)
    runs={name:inspect_run(getattr(args,name),eval_only=name=='r0') for name in ('r0','p1','old_p2','new_p2')}
    report=dict(version='gpu_repeat_json_audit_v1',verification='ANALYZED',created_at=utc(),runs=runs)
    old,new=runs['old_p2']['evaluation'],runs['new_p2']['evaluation']
    if old and new:
        diff={k:100*(new['metrics'][k]-old['metrics'][k]) for k in ('AP','APvt')}
        report['repeat_deltas_pp']=diff
        report['review_signal']=any(abs(v)>.3 for v in diff.values())
    lines=['# GPU3重复实验核验','', '## Material Passport','',
           '- 技能：academic-research-suite / experiment-agent；diagnosing-bugs。',
           '- 状态：ANALYZED；JSON离线核验，不代表受控测速已执行。','',
           '| 实验 | 状态 | AP | AP75 | APvt | 纯训练小时 | 评估小时 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,r in runs.items():
        e=r['evaluation']; m=e['metrics'] if e else {}
        values=[f'{100*m[k]:.3f}' if k in m else '待完成' for k in ('AP','AP75','APvt')]
        duration=f'{r.get("training_seconds",0)/3600:.3f}' if r['training'] else '—'
        et=f'{e["seconds"]/3600:.3f}' if e and e['seconds'] is not None else '待完成'
        lines.append(f'| {name} | {r["status"]} | '+ ' | '.join(values)+f' | {duration} | {et} |')
    lines+=['','GPU3任务未完整结束时，不能将表中局部耗时当作完整训练结果。',
            '本工具不读取pickle；八类PR/面积数组、ID集合、checkpoint内部恢复状态与进程退出需另行核验。',
            '旧P2慢速保留资源竞争嫌疑标记；跨GPU训练时间不用于质量loss成本判断。']
    save(root/'results.json',report)
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(root)


if __name__=='__main__': main()
