"""One-shot CPU analysis of two saved prediction exports; never imports a model."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.paired_head_errors import (analyze_image, add_pair, gt_groups, new_group,
                                    MODES, FP_TYPES, area_bucket, density_bucket)
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def resolve_export(path):
    if (path/'predictions.json').is_file():
        return path
    found = list(path.glob('predictions_*/predictions.json'))
    if len(found) != 1:
        raise ValueError(f'Expected exactly one export in {path}')
    return found[0].parent


def checked_box(box):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or (values[2:] <= 0).any():
        raise ValueError('Invalid xywh box; no clipping/removal is allowed')


def validate_export(path, ids, categories, image_info):
    meta = read(path/'metadata.json')
    if {c['id']:c['name'] for c in meta['categories']} != categories:
        raise ValueError('Category mapping differs from annotation')
    if len(meta['image_ids']) != len(ids) or set(meta['image_ids']) != set(ids):
        raise ValueError('Export metadata image coverage mismatch')
    rows = [json.loads(x) for x in (path/'images.jsonl').read_text().splitlines()]
    actual = [r['image_id'] for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(ids):
        raise ValueError('Duplicate/missing/unknown image IDs')
    diagnostics = {r['image_id']:r for r in rows}
    for r in rows:
        if r['query_mask_count'] != r['executed_query_count']:
            raise ValueError('Query mask mismatch')
        if (r['width'],r['height']) != (image_info[r['image_id']]['width'],image_info[r['image_id']]['height']):
            raise ValueError('Image coordinate dimensions mismatch')
    grouped = defaultdict(list)
    for record in read(path/'predictions.json'):
        if record['image_id'] not in diagnostics or record['category_id'] not in categories:
            raise ValueError('Unknown prediction image/category')
        checked_box(record['bbox'])
        if not np.isfinite(record['score']) or not 0 <= record['score'] <= 1:
            raise ValueError('Invalid prediction score')
        grouped[record['image_id']].append(record)
    for image_id in ids:
        if len(grouped[image_id]) != diagnostics[image_id]['executed_query_count']:
            raise ValueError('Export count differs from actual K; this protocol requires unfiltered K export')
    return grouped, diagnostics


def render_cases(output, cases, images, image_root):
    from PIL import Image, ImageDraw
    hashes = {}
    folder = output/'cases'
    folder.mkdir()
    for case in cases:
        gt = case['gt']
        source = (image_root/images[case['image_id']]['file_name']).resolve()
        if not source.is_relative_to(image_root.resolve()):
            raise ValueError('Image filename escapes image root')
        hashes[str(source)] = sha256(source)
        with Image.open(source) as opened:
            original = opened.convert('RGB')
        x,y,w,h = gt['bbox']
        pad = max(40, 2*max(w,h))
        crop = (max(0,int(x-pad)),max(0,int(y-pad)),
                min(original.width,int(x+w+pad+1)),min(original.height,int(y+h+pad+1)))
        panel = Image.new('RGB',(1200,690),'white')
        draw = ImageDraw.Draw(panel)
        draw.text((10,8),f"{case['kind']} | image={case['image_id']} GT={gt['id']} class={gt['category_id']}",fill='black')
        for col, version in enumerate(('epoch0','epoch1')):
            canvas = original.crop(crop).resize((600,600))
            painter = ImageDraw.Draw(canvas)
            sx,sy = 600/(crop[2]-crop[0]),600/(crop[3]-crop[1])
            def box(b, color, width):
                a,c,bw,bh=b
                painter.rectangle(((a-crop[0])*sx,(c-crop[1])*sy,(a+bw-crop[0])*sx,(c+bh-crop[1])*sy),outline=color,width=width)
            # Focal GT only: do not imply that other surrounding objects are unlabeled.
            for label,color,width in [('neighbor','orange',2),('best','cyan',3)]:
                p=case[version].get(label)
                if p is not None: box(p['bbox'],color,width)
            box(gt['bbox'],'red',2)
            panel.paste(canvas,(col*600,70))
            f=case[version]['features']
            draw.text((col*600+10,30),f"{version} best IoU={f['best_same_iou']:.4f} score={f['best_same_score']} rank={f['best_same_rank']}",fill='black')
            draw.text((col*600+10,48),'red: focal GT | cyan: best same-class IoU | orange: top-score neighbor',fill='black')
        case['file']=str(folder/f"{case['kind']}_{case['image_id']}_{gt['id']}.png")
        panel.save(case['file'])
    return hashes


def run(args):
    started=time.time()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Output already exists; use a new directory')
    exports=[resolve_export(p.resolve()) for p in [args.before,args.after]]
    inputs=[args.annotations.resolve(),args.image_ids.resolve()]
    inputs += [p/f for p in exports for f in ('predictions.json','images.jsonl','metadata.json')]
    inputs += [Path(__file__).resolve(), ROOT/'util/paired_head_errors.py', ROOT/'util/error_analysis.py']
    hashes={str(p):sha256(p) for p in inputs}
    ann=read(args.annotations)
    ids=read(args.image_ids)
    if not isinstance(ids,list) or any(type(i) is not int for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Image IDs must be unique integers')
    if args.expected_images is not None and len(ids)!=args.expected_images:
        raise ValueError('Unexpected subset size')
    image_info={r['id']:r for r in ann['images']}
    if len(image_info)!=len(ann['images']) or not set(ids)<=set(image_info):
        raise ValueError('Annotation image ID mismatch')
    categories={c['id']:c['name'] for c in ann['categories']}
    if len(categories)!=len(ann['categories']):
        raise ValueError('Duplicate categories')
    id_set=set(ids)
    ground_truth=defaultdict(list)
    seen=set()
    for g in ann['annotations']:
        if g['image_id'] not in id_set:
            continue
        if g['id'] in seen:
            raise ValueError('Duplicate GT annotation ID')
        seen.add(g['id'])
        if g.get('ignore',0) or g.get('iscrowd',0):
            raise ValueError('This paired protocol excludes no GT silently; ignore/crowd requires separate protocol')
        checked_box(g['bbox'])
        if g['category_id'] not in categories or not np.isfinite(g.get('area',0)) or g.get('area',0)<0:
            raise ValueError('Invalid GT category/area')
        ground_truth[g['image_id']].append(g)
    if args.expected_gt is not None and len(seen)!=args.expected_gt:
        raise ValueError('Unexpected GT support')
    pairs=[validate_export(p,ids,categories,image_info) for p in exports]
    for image_id in ids:
        if any(d[image_id]['gt_count']!=len(ground_truth[image_id]) for _,d in pairs):
            raise ValueError('Diagnostic GT count mismatch')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(version='paired_head_errors_v1',started_at=datetime.now().astimezone().isoformat(),
        command=sys.argv,inputs_sha256=hashes,python=platform.python_version(),numpy=np.__version__,
        modes=MODES,thresholds=[.1,.5,.75],score_threshold=.25,top_limit=300,
        images=len(ids),gt=len(seen),scope='exported detections only; no model execution; engineering diagnosis',
        coordinate_policy='original-pixel xywh; no clipping; strictly positive dimensions',
        tie_policy='score descending / original export order; GT IoU tie -> smallest annotation ID',
        fp_size_policy='associated GT diagnostic size; background=unassigned; class=predicted class')
    write(output/'manifest.json',manifest)
    groups=defaultdict(new_group)
    pred_groups={v:defaultdict(Counter) for v in ('epoch0','epoch1')}
    candidates={k:{} for k in ('localization_drop','ranking_issue','improvement')}
    per_image=[]
    with (output/'paired_gt.jsonl').open('w',encoding='utf-8') as gt_stream, (output/'prediction_errors.jsonl').open('w',encoding='utf-8') as pred_stream:
        for number,image_id in enumerate(sorted(ids),1):
            original_k=int(pairs[0][1][image_id]['raw_query_counts'])
            source_gt=ground_truth[image_id]
            versions=[analyze_image(p[0][image_id],source_gt) for p in pairs]
            image_row=dict(image_id=image_id,gt=len(source_gt),original_K=original_k)
            for version, (gt,features,labels) in zip(('epoch0','epoch1'),versions):
                by_id={g['id']:g for g in gt}
                mode_counts={m:Counter() for m in MODES}
                for p in labels:
                    for mode,label in p['modes'].items():
                        associated=by_id.get(label['associated_gt_id'])
                        size=area_bucket(associated.get('area',associated['bbox'][2]*associated['bbox'][3])) if associated else 'unassigned'
                        buckets=['overall','class:'+str(p['category_id']), 'size:'+size,
                                 'density:'+density_bucket(len(gt)), 'original_K:'+str(original_k)]
                        mode_counts[mode][label['label']]+=1
                        for bucket in buckets:
                            pred_groups[version][bucket][mode+'_'+label['label']]+=1
                    line(pred_stream,dict(image_id=image_id,version=version,**p))
                image_row[version]=mode_counts
            per_image.append(image_row)
            for g,b,a in zip(versions[0][0],versions[0][1],versions[1][1]):
                row=dict(image_id=image_id,annotation_id=g['id'],category_id=g['category_id'],
                         bbox=g['bbox'],area=g.get('area',g['bbox'][2]*g['bbox'][3]),original_K=original_k,
                         epoch0=b,epoch1=a)
                line(gt_stream,row)
                for bucket in gt_groups(g,len(source_gt),original_k):
                    add_pair(groups[bucket],b,a)
                delta=a['best_same_iou']-b['best_same_iou']
                scores={}
                if delta<0: scores['localization_drop']=-delta
                if delta>0: scores['improvement']=delta
                if a['thresholds']['75']['ranking_inconsistent']:
                    scores['ranking_issue']=a['best_same_iou']-a['neighbor_iou']
                for kind,score in scores.items():
                    old=candidates[kind].get(image_id)
                    if old and (old['change_score']>score or (old['change_score']==score and old['gt']['id']<g['id'])):
                        continue
                    case=dict(kind=kind,change_score=score,image_id=image_id,gt=g)
                    for version,f,p in [('epoch0',b,pairs[0]),('epoch1',a,pairs[1])]:
                        records=p[0][image_id]
                        case[version]=dict(features=f,
                            best=records[f['best_same_prediction_index']] if f['best_same_prediction_index'] is not None else None,
                            neighbor=records[f['neighbor_prediction_index']] if f['neighbor_prediction_index'] is not None else None)
                    candidates[kind][image_id]=case
            if number%100==0:
                print(f'Analyzed {number}/{len(ids)} images',flush=True)
    for bucket,g in groups.items():
        g['low_support']=g['gt']<100
        g['images']=sum(1 for r in per_image if bucket=='overall' or
            bucket=='density:'+density_bucket(r['gt']) or bucket=='original_K:'+str(r['original_K'])) if not bucket.startswith(('class:','size:')) else None
        for version in ('epoch0','epoch1'):
            for mode in MODES:
                assert g[version][mode+'_tp']+g[version][mode+'_fn']==g['gt']
        assert sum(map(sum,g['transitions']))==g['gt']
    chosen=[]
    used=set()
    for kind in candidates:
        count=0
        for case in sorted(candidates[kind].values(),key=lambda c:(-c['change_score'],c['image_id'],c['gt']['id'])):
            if case['image_id'] in used: continue
            chosen.append(case);used.add(case['image_id']);count+=1
            if count==4: break
    if args.image_root:
        hashes.update(render_cases(output,chosen,image_info,args.image_root))
    else:
        chosen=[]
    write(output/'cases.json',chosen)
    summary=dict(images=len(ids),gt=len(seen),categories=categories,groups=groups,
        prediction_groups=pred_groups,per_image=per_image,
        warning='GT coverage is many-to-one; TP/FP/FN are diagnostic one-to-one; no AP recomputation')
    write(output/'summary.json',summary)
    after={p:sha256(Path(p)) for p in hashes}
    if after!=hashes:
        raise ValueError('Input/source hashes changed during analysis')
    manifest.update(inputs_sha256=hashes,inputs_unchanged=True,status='completed',seconds=time.time()-started,
                    outputs_sha256={p.name:sha256(p) for p in output.glob('*.jsonl')})
    write(output/'manifest.json',manifest)
    print(f'Complete: {len(ids)} images / {len(seen)} GT in {time.time()-started:.1f}s -> {output}',flush=True)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--image-ids',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-root',type=Path)
    p.add_argument('--expected-images',type=int)
    p.add_argument('--expected-gt',type=int)
    return p


if __name__=='__main__':
    run(parser().parse_args())
