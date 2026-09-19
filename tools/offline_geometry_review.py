"""Read-only CPU geometry review of frozen D0 predictions. No torch/model imports."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid
import numpy as np

EXPECTED = {
    'annotation': '9817a75f9bc4a84015881f2ddbf39bcc29635654a0266b210fd382743c927d98',
    'image_ids': 'ca8ea6d5b2db447a466bcba7ae0f768ec1fb420a33a8d6c2b87f6543f2ce9182',
    'precheck_ids': '29bf79c41a519d7acc059522b499ece6d4eeb3207eba9837f8862acbad08a139',
    'predictions': '7ac34235f5d492de13bcab1b4e5d24093f232f16e143c6d97e395dd6dd2ab323',
}
CHECKPOINT = 'c447367e2d19411319a990985e0127af8db08bfb7eec0329b2065b095ccc64eb'
IMAGE_COUNT, PRECHECK_COUNT, GT_COUNT = 1000, 32, 90536

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def validate_ids(ids, population, count):
    if len(ids)!=count or any(type(i) is not int for i in ids) or len(set(ids))!=count or not set(ids)<=set(population):
        raise ValueError('Duplicate, unknown or incomplete image IDs')

def validate_box(box):
    a=np.asarray(box, dtype=float)
    if a.shape!=(4,) or not np.isfinite(a).all() or (a[2:]<=0).any():
        raise ValueError('Invalid pixel xywh box')

def best_matches(predictions, gt, same_class=False, chunk=256):
    """First exported index wins exact IoU ties; no confidence threshold or clipping."""
    if chunk<1: raise ValueError('chunk must be positive')
    truth=np.asarray([g['bbox'] for g in gt],float).reshape(-1,4)
    best=np.zeros(len(gt)); indices=np.full(len(gt),-1,dtype=int)
    cats=np.asarray([g['category_id'] for g in gt])
    for start in range(0,len(predictions),chunk):
        batch=predictions[start:start+chunk]
        box=np.asarray([p['bbox'] for p in batch],float).reshape(-1,4)
        if not len(gt): break
        lo=np.maximum(box[:,None,:2],truth[None,:,:2])
        hi=np.minimum(box[:,None,:2]+box[:,None,2:],truth[None,:,:2]+truth[None,:,2:])
        inter=np.maximum(hi-lo,0).prod(-1)
        overlap=inter/np.maximum(box[:,2:].prod(-1)[:,None]+truth[:,2:].prod(-1)[None]-inter,1e-12)
        if same_class:
            overlap=np.where(np.asarray([p['category_id'] for p in batch])[:,None]==cats[None],overlap,0)
        local=overlap.argmax(0); values=overlap[local,np.arange(len(gt))]
        update=values>best
        best[update]=values[update]; indices[update]=start+local[update]
    result=[]
    for j,g in enumerate(gt):
        index=int(indices[j]); row=dict(best_iou=float(best[j]),prediction_index=index if index>=0 else None,
                                      prediction_class=None,score=None,error=None)
        if index>=0:
            p=predictions[index]; row.update(prediction_class=p['category_id'],score=p['score'])
            if best[j]>=.1:
                b=np.asarray(p['bbox'],float); t=truth[j]
                center=(b[:2]+b[2:]/2)-(t[:2]+t[2:]/2)
                row['error']=dict(dx_px=float(center[0]),dy_px=float(center[1]),
                    dx_relative=float(center[0]/t[2]),dy_relative=float(center[1]/t[3]),
                    width_ratio=float(b[2]/t[2]),height_ratio=float(b[3]/t[3]))
        result.append(row)
    return result

def stats(values):
    x=np.asarray(values,float)
    if not len(x): return dict(n=0,mean=None,median=None,p10=None,p90=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))

def aggregate(rows, mode):
    records=[r[mode] for r in rows]; n=len(rows)
    result=dict(gt=n,images_with_gt=len({r['image_id'] for r in rows}),low_support=n<100,
                hit50=sum(r['best_iou']>=.5 for r in records),hit75=sum(r['best_iou']>=.75 for r in records),
                no_neighbor=sum(r['best_iou']<.1 for r in records))
    for threshold in (.1,.5):
        selected=[(r['image_id'],r[mode]['error']) for r in rows if r[mode]['best_iou']>=threshold]
        errors=[v for _,v in selected]
        measures={}
        for key in ('dx_px','dy_px','dx_relative','dy_relative','width_ratio','height_ratio'):
            measures[key]=stats([e[key] for e in errors])
            if key.startswith(('dx','dy')): measures['abs_'+key]=stats([abs(e[key]) for e in errors])
        per_image=defaultdict(list)
        for image_id,e in selected: per_image[image_id].append(e)
        measures['image_balanced_medians']={key:stats([float(np.median([e[key] for e in es])) for es in per_image.values()])
            for key in ('width_ratio','height_ratio')}
        measures['both_dimensions_oversized_fraction']=(sum(e['width_ratio']>1 and e['height_ratio']>1 for e in errors)/len(errors) if errors else None)
        result['neighbor_iou_ge_'+str(threshold)]=dict(gt_support=len(errors),image_support=len(per_image),measures=measures)
    return result

def summarize(rows, categories):
    groups=defaultdict(list)
    for c in categories: groups['category/'+str(c)]  # Preserve absent categories.
    for r in rows:
        area=r['area']; size='verytiny' if area<64 else 'tiny' if area<256 else 'small' if area<1024 else 'medium'
        count=r['image_gt_count']; density='0' if count==0 else '1-100' if count<=100 else '101-300' if count<=300 else '301-900' if count<=900 else '>900'
        cohort='original32' if r['in_precheck32'] else 'remaining968'
        keys=['all','category/'+str(r['category_id']),'size/'+size,'density/'+density,'query/'+str(r['query_count']),
              'category_size/'+str(r['category_id'])+'/'+size,'cohort/'+cohort,'cohort_size/'+cohort+'/'+size]
        for key in keys: groups[key].append(r)
    return {key:{mode:aggregate(rs,mode) for mode in ('any_class','same_class')} for key,rs in groups.items()}

def execute(a,out,log):
    candidates=list(a.run_dir.glob('predictions_*/predictions.json'))
    if len(candidates)!=1: raise ValueError('Expected exactly one D0 prediction export')
    pred_dir=candidates[0].parent
    paths=dict(annotation=a.annotation,image_ids=a.image_ids,precheck_ids=a.precheck_ids,predictions=candidates[0],
               images=pred_dir/'images.jsonl',metadata=pred_dir/'metadata.json',
               diagnosis_manifest=a.run_dir/'diagnosis_manifest.json',diagnosis_status=a.run_dir/'diagnosis_status.json',
               source=Path(__file__))
    hashes={k:digest(p) for k,p in paths.items()}
    for k,v in EXPECTED.items():
        if hashes[k]!=v: raise ValueError('Frozen input SHA-256 differs: '+k)
    manifest=read(paths['diagnosis_manifest']); status=read(paths['diagnosis_status'])
    if manifest['checkpoint_sha256']!=CHECKPOINT or manifest['group']!='D0' or status['status']!='completed' or status['images']!=IMAGE_COUNT:
        raise ValueError('Expected completed final-epoch D0 diagnosis')
    if manifest['annotation_sha256']!=hashes['annotation'] or manifest['subset_sha256']!=hashes['image_ids']:
        raise ValueError('Diagnosis/input provenance differs')
    ann=read(a.annotation); ids=read(a.image_ids); pre=read(a.precheck_ids)
    image_map={i['id']:i for i in ann['images']}
    if len(image_map)!=len(ann['images']): raise ValueError('Duplicate annotation image IDs')
    validate_ids(ids,image_map,IMAGE_COUNT); validate_ids(pre,ids,PRECHECK_COUNT)
    category_names={c['id']:c['name'] for c in ann['categories']}
    if len(category_names)!=8 or len(category_names)!=len(ann['categories']): raise ValueError('Expected eight original categories')
    metadata=read(paths['metadata']); validate_ids(metadata['image_ids'],ids,IMAGE_COUNT)
    if {c['id']:c['name'] for c in metadata['categories']}!=category_names: raise ValueError('Category mapping mismatch')
    image_records=[json.loads(line) for line in paths['images'].read_text().splitlines()]
    validate_ids([i['image_id'] for i in image_records],ids,IMAGE_COUNT)
    info={i['image_id']:i for i in image_records}
    gt=defaultdict(list); seen=set(); selected=set(ids)
    for g in ann['annotations']:
        if g['image_id'] not in selected: continue
        if g['id'] in seen or g['category_id'] not in category_names or g.get('ignore',0) or g.get('iscrowd',0):
            raise ValueError('Duplicate/invalid/ignored GT requires separate protocol')
        seen.add(g['id']); validate_box(g['bbox'])
        if not np.isfinite(g['area']) or g['area']<=0: raise ValueError('Invalid GT area')
        gt[g['image_id']].append(g)
    if len(seen)!=GT_COUNT: raise ValueError('Unexpected frozen GT count')
    pred=defaultdict(list)
    for p in read(paths['predictions']):
        if p['image_id'] not in selected or p['category_id'] not in category_names: raise ValueError('Unknown prediction image/category ID')
        validate_box(p['bbox'])
        if not np.isfinite(p['score']) or not 0<=p['score']<=1: raise ValueError('Invalid score')
        pred[p['image_id']].append(p)
    for i in ids:
        if info[i]['height']!=image_map[i]['height'] or info[i]['width']!=image_map[i]['width']: raise ValueError('Image dimensions differ')
        if info[i]['gt_count']!=len(gt[i]) or info[i]['executed_query_count']!=len(pred[i]): raise ValueError('GT/prediction coverage differs')
    save(out/'manifest.json',dict(version='offline_geometry_v1',command=sys.argv,input_paths={k:str(v) for k,v in paths.items()},
        input_sha256=hashes,checkpoint_sha256=CHECKPOINT,python=platform.python_version(),numpy=np.__version__,
        scope='Existing exported predictions only; no full queries, no new inference, no score filtering',
        geometry_condition='best IoU >= .10; sensitivity >= .50; first exported index wins ties',
        area_bins='[0,64),[64,256),[256,1024),[1024,infinity); diagnostic, not official protocol'))
    rows=[]
    with (out/'paired_gt.jsonl').open('x',encoding='utf-8') as f:
        for step,i in enumerate(ids,1):
            any_class=best_matches(pred[i],gt[i]); same_class=best_matches(pred[i],gt[i],True)
            for g,any_rec,same_rec in zip(gt[i],any_class,same_class):
                r=dict(image_id=i,annotation_id=g['id'],category_id=g['category_id'],area=g['area'],
                    image_gt_count=len(gt[i]),query_count=info[i]['executed_query_count'],in_precheck32=i in pre,
                    any_class=any_rec,same_class=same_rec)
                rows.append(r); f.write(json.dumps(r,allow_nan=False)+'\n')
            if step%100==0: log(f'[geometry] {step}/1000 images; GT={len(rows)}'); f.flush()
    assert len(rows)==GT_COUNT
    groups=summarize(rows,category_names)
    image_summary=[dict(image_id=i,gt=len(gt[i]),predictions=len(pred[i]),in_precheck32=i in pre,
                        query_count=info[i]['executed_query_count']) for i in ids]
    save(out/'images.json',image_summary)
    result=dict(images=IMAGE_COUNT,gt=len(rows),categories=category_names,strata=groups,
                empty_gt_images=sum(len(gt[i])==0 for i in ids),purpose='test engineering diagnosis; not AP or one-to-one TP')
    save(out/'summary.json',result)
    lines=['# 1000图导出框离线几何复核','', '## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent; Mode: validate preparation.',
        '- 状态：CPU确定性描述统计已完成；因果机制、研究效果未验证。','',
        '不运行模型，不计算新AP，不修改预测。覆盖允许一框对应多GT，不是TP。误差条件IoU≥0.10；同时保存≥0.50敏感性统计。', '',
        '只分析原生导出框，不能当作1000图全query证据；original32与remaining968只作描述比较。', '',
        '| 分层（任意类别） | GT | 邻近框GT | @.75覆盖数 | 水平绝对误差/GT宽均值 | 宽比中位数 | 高比中位数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,g in groups.items():
        if not (key=='all' or key.startswith(('category/','size/','density/','cohort_size/'))): continue
        s=g['any_class']; near=s['neighbor_iou_ge_0.1']; m=near['measures']
        def fmt(v): return '—' if v is None else f'{v:.4f}'
        lines.append(f"| {key} | {s['gt']} | {near['gt_support']} | {s['hit75']} | {fmt(m['abs_dx_relative']['mean'])} | {fmt(m['width_ratio']['median'])} | {fmt(m['height_ratio']['median'])} |")
    lines+=['','完整JSON保留同类框、分位数、图像平衡尺寸统计、全部类别及低支持标记。',
            '无邻近框不计算虚假的中心/宽高误差；空GT图像在images.json保留。GT数量并不等于独立样本数量。',
            '本工具不据此自动选择损失、缩框、训练或升级默认模型。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(digest(paths[k])!=v for k,v in hashes.items()): raise ValueError('Input changed during analysis')
    log('Completed: 1000 images, 90536 GT; inputs unchanged. No model was loaded.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run-dir','annotation','image-ids','precheck-ids','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    # One unique output for every manual invocation; never overwrite inputs/history.
    out=a.output_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    with (out/'console.log').open('x',encoding='utf-8') as f:
        def log(message): print(message,flush=True); f.write(message+'\n'); f.flush()
        log('Output: '+str(out)); log('CPU-only: reading frozen D0 exports; no training/inference.')
        try:
            execute(a,out,log)
        except BaseException as exc:
            log(traceback.format_exc())
            save(out/'run_status.json',dict(status='failed',error=repr(exc),elapsed_seconds=time.time()-start))
            raise
        save(out/'run_status.json',dict(status='completed',images=1000,gt=90536,elapsed_seconds=time.time()-start,input_hashes_unchanged=True))
        log(f'Elapsed: {time.time()-start:.2f}s')

if __name__=='__main__': main()
