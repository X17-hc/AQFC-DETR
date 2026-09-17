"""Read-only image/annotation audit; never repair, remove or substitute samples."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from PIL import Image


def audit(root, dataset, decode=True):
    root=Path(root)
    splits=('train','val','trainval','test') if dataset=='aitodv2' else ('train','val')
    report={'dataset':dataset,'root':str(root),'splits':{},'issues':[], 'status':'passed'}
    members={}
    image_members={}
    checked={}
    for split in splits:
        ann=(root/'annotations'/f'aitodv2_{split}.json' if dataset=='aitodv2' else
             root/'annotations_coco'/f'VisDrone2019-DET_{split}_coco.json')
        folder=(root/'images'/split/'images' if dataset=='aitodv2' else root/f'VisDrone2019-DET-{split}'/'images')
        if not ann.exists():
            report['issues'].append({'split':split,'error':'missing_annotation','path':str(ann)})
            continue
        raw=ann.read_bytes(); data=json.loads(raw)
        images=data['images']; labels=data['annotations']
        ids=[x['id'] for x in images]; members[split]=set(ids)
        # COCO IDs are local to each annotation file, not global image identities.
        names=[Path(x['file_name'].replace('\\', '/')).as_posix() for x in images]
        image_members[split]=set(names)
        duplicate_names=[name for name,n in Counter(names).items() if n>1]
        if duplicate_names:
            report['issues'].append(dict(split=split,error='duplicate_file_name',names=duplicate_names))
        valid_categories=set(range(8) if dataset=='aitodv2' else range(1,11))
        categories={x['id'] for x in data['categories']}
        if categories!=valid_categories:
            report['issues'].append(dict(split=split,error='category_contract',categories=sorted(categories)))
        for field,items in [('image_id',ids),('annotation_id',[x['id'] for x in labels])]:
            duplicates=[i for i,n in Counter(items).items() if n>1]
            if duplicates: report['issues'].append(dict(split=split,error='duplicate_'+field,ids=duplicates))
        for obj in labels:
            box=obj.get('bbox',[])
            ignored=bool(obj.get('ignore') or obj.get('iscrowd'))
            if obj['image_id'] not in members[split] or (obj['category_id'] not in valid_categories and not (dataset=='visdrone' and ignored)):
                report['issues'].append(dict(split=split,error='invalid_annotation_reference',id=obj['id']))
            if len(box)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
                report['issues'].append(dict(split=split,error='malformed_or_nonfinite_box',id=obj['id']))
            elif box[2]<=0 or box[3]<=0:
                report['issues'].append(dict(split=split,error='invalid_box',id=obj['id'],severity='warning',
                    reason='recorded; existing loader filters zero/negative-area GT'))
        for item in images:
            path=(folder/item['file_name']).resolve()
            if not path.is_relative_to(folder.resolve()):
                report['issues'].append(dict(split=split,error='unsafe_image_path',image_id=item['id']))
                continue
            try:
                key=str(path)
                if key not in checked:
                    with Image.open(path) as im:
                        size=im.size
                        if decode: im.load()
                    checked[key]=size
                if checked[key]!=(item['width'],item['height']):
                    raise ValueError('Decoded size differs from annotation')
            except (OSError,ValueError) as exc:
                report['issues'].append(dict(split=split,error='unreadable_or_size_mismatch',image_id=item['id'],path=str(path),detail=str(exc)))
        report['splits'][split]=dict(images=len(images),annotations=len(labels),annotation_sha256=hashlib.sha256(raw).hexdigest(),decoded=decode)
        print(f'[Data audit] {dataset}/{split}: {len(images)} images checked',flush=True)
    if dataset=='aitodv2' and set(splits)<=set(members):
        members=image_members
        report['split_identity_basis']='normalized file_name; COCO IDs are split-local; not a pixel-content duplicate audit'
        report['split_relationships']={'train_val_overlap':len(members['train']&members['val']),
            'trainval_test_overlap':len(members['trainval']&members['test']),
            'trainval_equals_union':members['trainval']==members['train']|members['val']}
        if report['split_relationships']!={'train_val_overlap':0,'trainval_test_overlap':0,'trainval_equals_union':True}:
            report['issues'].append(dict(error='split_membership_contract'))
    report['status']='blocked' if any(x.get('severity')!='warning' for x in report['issues']) else 'passed'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--dataset',choices=['aitodv2','visdrone'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    result=audit(args.data_root,args.dataset)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'issues':len(result['issues'])}))
    raise SystemExit(0 if result['status']=='passed' else 2)


if __name__=='__main__': main()
