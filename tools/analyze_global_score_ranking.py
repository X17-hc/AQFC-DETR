"""Official AI-TOD AP75 matching plus descriptive within-class, cross-image ranks.

Consumes existing JSON predictions only; never loads a checkpoint or model.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys
import time
import inspect
import copy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_paired_head_errors import read, write, sha256, resolve_export, validate_export, line
from aitodpycocotools.coco import COCO
from aitodpycocotools.cocoeval import COCOeval

EDGES=[0,.05,.1,.25,.5,.75,1.0000000001]
LABELS=['tp','duplicate','class_confusion','localization','class_and_localization','background']


def rank_curve(scores, matched, ignored, n_gt):
    """Mirror COCO accumulation order/interpolation without altering matching."""
    order=np.argsort(-np.asarray(scores),kind='mergesort')
    matched=np.asarray(matched,dtype=bool)[order]
    ignored=np.asarray(ignored,dtype=bool)[order]
    tp=np.cumsum(matched & ~ignored).astype(float)
    fp=np.cumsum(~matched & ~ignored).astype(float)
    precision=tp/(tp+fp+np.spacing(1))
    recall=tp/n_gt if n_gt else np.zeros_like(tp)
    envelope=np.maximum.accumulate(precision[::-1])[::-1]
    q=np.zeros(101)
    positions=np.searchsorted(recall,np.linspace(0,1,101),side='left')
    valid=positions<len(recall)
    q[valid]=envelope[positions[valid]]
    return dict(order=order,tp=tp,fp=fp,precision=precision,recall=recall,q=q,
                AP75=float(q.mean()) if n_gt else None,
                max_recall=float(recall[-1]) if n_gt and len(recall) else 0.)


def official_eval(annotation,records,ids):
    gt=COCO()
    # _prepare/loadRes mutate annotation dictionaries; isolate inputs in memory.
    gt.dataset=copy.deepcopy(annotation);gt.createIndex()
    if records:
        dt=gt.loadRes(copy.deepcopy(records))
    else:
        dt=COCO();dt.dataset=dict(images=copy.deepcopy(annotation['images']),
            categories=copy.deepcopy(annotation['categories']),annotations=[]);dt.createIndex()
    evaluator=COCOeval(gt,dt,'bbox')
    evaluator.params.imgIds=sorted(ids)
    evaluator.params.catIds=sorted(gt.getCatIds())
    evaluator.params.iouThrs=np.array([.75])
    evaluator.params.areaRng=[[0,1e10]]
    evaluator.params.areaRngLbl=['all']
    evaluator.params.maxDets=[1500]
    evaluator.params.useCats=1
    evaluator.evaluate();evaluator.accumulate(with_lrp=False)
    return evaluator


def diagnostic_labels(path,version,grouped):
    codes={i:np.full(len(p),255,dtype=np.uint8) for i,p in grouped.items()}
    with path.open(encoding='utf-8') as f:
        for text in f:
            r=json.loads(text)
            if r['version']!=version: continue
            image,index=r['image_id'],r['prediction_index']
            if image not in codes or not 0<=index<len(codes[image]) or codes[image][index]!=255:
                raise ValueError('Duplicate or unknown prior diagnostic prediction')
            codes[image][index]=LABELS.index(r['modes']['all_75']['label'])
    if any(np.any(c==255) for c in codes.values()):
        raise ValueError('Incomplete diagnostic-label join')
    return codes


def class_ranking(evaluator,category_index,records,local_indices,codes,stream):
    category=evaluator.params.catIds[category_index]
    entries=[e for e in evaluator.evalImgs if e is not None and e['category_id']==category]
    scores=np.concatenate([e['dtScores'] for e in entries]) if entries else np.array([])
    matches=np.concatenate([e['dtMatches'][0] for e in entries]) if entries else np.array([])
    ignores=np.concatenate([e['dtIgnore'][0] for e in entries]) if entries else np.array([],dtype=bool)
    ids=np.concatenate([e['dtIds'] for e in entries]).astype(int) if entries else np.array([],dtype=int)
    n_gt=sum(np.count_nonzero(e['gtIgnore']==0) for e in entries)
    curve=rank_curve(scores,matches,ignores,n_gt)
    official=evaluator.eval['precision'][0,:,category_index,0,0]
    if n_gt:
        np.testing.assert_allclose(curve['q'],official,atol=1e-12,rtol=0)
    else:
        assert (official==-1).all()
    order=curve['order'];scores=scores[order];matches=matches[order];ignores=ignores[order];ids=ids[order]
    bins=[Counter() for _ in range(len(EDGES)-1)]
    thresholds={str(t):Counter() for t in [.1,.25,.5,.75]}
    overall=Counter();tp_ranks=[];fp_before_tp=[]
    for rank,(score,match,ignore,dt_id) in enumerate(zip(scores,matches,ignores,ids),1):
        p=records[dt_id-1];local_index=local_indices[dt_id-1]
        prior=LABELS[int(codes[p['image_id']][local_index])]
        label='ignored' if ignore else 'tp' if match else 'protocol_matching_difference' if prior=='tp' else prior
        overall[label]+=1
        bins[min(len(bins)-1,int(np.searchsorted(EDGES,score,side='right')-1))][label]+=1
        for t,counts in thresholds.items():
            if score>=float(t): counts[label]+=1
        if label=='tp':
            tp_ranks.append(rank);fp_before_tp.append(int(curve['fp'][rank-1]))
        line(stream,dict(category_id=int(category),image_id=p['image_id'],prediction_index=local_index,
             class_global_rank=rank,score=float(score),matched_gt_id=int(match) if match else None,
             ignored=bool(ignore),official_status='ignored' if ignore else 'tp' if match else 'fp',
             diagnostic_subtype=label,prior_diagnostic_label=prior,
             cumulative_tp=int(curve['tp'][rank-1]),cumulative_fp=int(curve['fp'][rank-1]),
             precision=float(curve['precision'][rank-1]),recall=float(curve['recall'][rank-1])))
    recall_points=[]
    for target in np.arange(.05,1.001,.05):
        target=float(round(target,2));position=int(np.searchsorted(curve['recall'],target,side='left'))
        if position>=len(scores) or n_gt==0:
            recall_points.append(dict(target=target,reachable=False))
        else:
            recall_points.append(dict(target=target,reachable=True,rank=position+1,score=float(scores[position]),
                tp=int(curve['tp'][position]),fp=int(curve['fp'][position]),
                precision=float(curve['precision'][position]),actual_recall=float(curve['recall'][position])))
    def distribution(values):
        return dict(n=len(values),median=float(np.median(values)),p90=float(np.percentile(values,90))) if values else dict(n=0,median=None,p90=None)
    return dict(category_id=int(category),gt=int(n_gt),low_support=n_gt<100,
        AP75=curve['AP75'],max_recall=curve['max_recall'],precision101=curve['q'].tolist() if n_gt else None,
        ranked_predictions=len(scores),counts=overall,score_bins=bins,cumulative_thresholds=thresholds,
        tp_rank_distribution=distribution(tp_ranks),fp_before_tp_distribution=distribution(fp_before_tp),
        recall_points=recall_points,official_precision_reconstruction_passed=True)


def run(args):
    output=args.output.resolve()
    if output.exists(): raise FileExistsError('Use a new output directory')
    start=time.time()
    exports=[resolve_export(p) for p in [args.before,args.after]]
    input_paths=[args.annotations,args.image_ids,args.diagnostic_errors,args.before/'log.txt',args.after/'log.txt']
    input_paths += [e/n for e in exports for n in ['predictions.json','images.jsonl','metadata.json']]
    input_paths += [Path(__file__),Path(inspect.getfile(COCOeval)),Path(inspect.getfile(COCO)),
                    ROOT/'tools/analyze_paired_head_errors.py']
    hashes={str(p.resolve()):sha256(p) for p in input_paths}
    annotation=read(args.annotations);ids=read(args.image_ids)
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate subset IDs')
    categories={c['id']:c['name'] for c in annotation['categories']}
    image_info={i['id']:i for i in annotation['images']}
    versions={}
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='started',command=sys.argv,inputs_sha256=hashes,
        protocol=dict(iou=.75,area=[0,1e10],maxDets=1500,useCats=True,
            score_edges=EDGES,recall_points='0.05 to 1.00 by 0.05',with_lrp=False),
        purpose='engineering_check; saved predictions only; no model inference')
    write(output/'manifest.json',manifest)
    for name,export,original in zip(['epoch0','epoch1'],exports,[args.before,args.after]):
        grouped,_=validate_export(export,ids,categories,image_info)
        codes=diagnostic_labels(args.diagnostic_errors,name,grouped)
        records=read(export/'predictions.json')
        counts=Counter();local_indices=[]
        for p in records:
            local_indices.append(counts[p['image_id']]);counts[p['image_id']]+=1
        evaluator=official_eval(annotation,records,ids)
        classes=[]
        with (output/(name+'_global_ranks.jsonl')).open('w',encoding='utf-8') as stream:
            for index,category in enumerate(evaluator.params.catIds):
                c=class_ranking(evaluator,index,records,local_indices,codes,stream)
                c['name']=categories[category];classes.append(c)
        macro=float(np.mean([c['AP75'] for c in classes if c['AP75'] is not None]))
        historical=json.loads((original/'log.txt').read_text().splitlines()[-1])['test_bbox_metrics']['AP75']
        difference=macro-historical
        versions[name]=dict(AP75=macro,historical_AP75=historical,difference=difference,classes=classes)
        write(output/(name+'_summary.json'),versions[name])
        if abs(difference)>1e-6:
            raise ValueError(f'{name} historical AP75 mismatch: {difference}; no interpretation')
        print(f'{name}: AP75={macro:.12f}, historical difference={difference:.3g}',flush=True)
        del evaluator,grouped,codes,records
    n=sum(c['AP75'] is not None for c in versions['epoch0']['classes'])
    attribution=[]
    for b,a in zip(versions['epoch0']['classes'],versions['epoch1']['classes']):
        if (b['gt']==0)!=(a['gt']==0): raise ValueError('Different valid class sets')
        attribution.append(dict(name=b['name'],category_id=b['category_id'],gt=b['gt'],
            AP75_delta=a['AP75']-b['AP75'] if b['gt'] else None,
            macro_delta_contribution=(a['AP75']-b['AP75'])/n if b['gt'] else None,
            shared_recall=[dict(target=x['target'],epoch0=x,epoch1=y) for x,y in zip(b['recall_points'],a['recall_points']) if x['reachable'] and y['reachable']]))
    write(output/'comparison.json',dict(versions=versions,attribution=attribution,
        warning='Per-class delta/n is arithmetic decomposition, not model-module causal attribution. FP subtypes are prior diagnostic labels; official matching controls TP/FP.'))
    unchanged=all(sha256(Path(p))==h for p,h in hashes.items())
    if not unchanged: raise ValueError('Inputs changed during analysis')
    manifest.update(status='completed',seconds=time.time()-start,inputs_unchanged=True,
        evaluator_precision_reconstruction_atol=1e-12,historical_AP75_tolerance=1e-6,
        outputs_sha256={p.name:sha256(p) for p in output.glob('*.jsonl')})
    write(output/'manifest.json',manifest)
    print(f'Completed in {time.time()-start:.1f}s: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['before','after','annotations','image-ids','diagnostic-errors','output']:
        p.add_argument('--'+arg,type=Path,required=True)
    run(p.parse_args())
