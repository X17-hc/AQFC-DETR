"""MANUAL evaluation-only 32-image export check followed by a fixed 1000-image pair."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime,timezone
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def run(cli,out):
    import numpy as np
    import torch
    from main import build_model_main,build_data_loaders
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    from util.dome_transfer import initialize,underestimate_loss,SOURCE_SHA
    from util.experiment import sha256
    from tools.legacy24_diagnosis import proposal_record
    from tools.diagnose_f2_boxes import paired_results
    from util.factor_diagnostics import export_class_metrics,finite_json
    from models.aqfcdetr.query_allocator import QueryBudgetLoss
    from datasets import get_coco_api_from_dataset
    from datasets.coco_eval import CocoEvaluator
    from tools.benchmark_interleaved import source_hashes
    annotation=Path(cli.data_root)/'annotations/aitodv2_test.json'
    source_config=Path(cli.pretrained).parent/'config_args_all.json'
    inputs=[Path(cli.pretrained),source_config,Path(cli.precheck_ids),Path(cli.image_ids),annotation]
    hashes={str(p):sha256(p) for p in inputs}
    if hashes[str(Path(cli.pretrained))]!=SOURCE_SHA:raise ValueError('Unexpected initialization')
    ann=json.loads(annotation.read_text());images={i['id']:i for i in ann['images']}
    gt={i:[] for i in images}
    seen_gt=set()
    for g in ann['annotations']:
        if g['id'] in seen_gt:raise ValueError('Duplicate annotation ID')
        seen_gt.add(g['id'])
        if not g.get('ignore',0) and not g.get('iscrowd',0):gt[g['image_id']].append(g)
    lists={}
    for key,path,n in [('precheck32',cli.precheck_ids,32),('subset1000',cli.image_ids,1000)]:
        ids=json.loads(Path(path).read_text())
        if len(ids)!=n or len(set(ids))!=n or not set(ids)<=set(images):raise ValueError('Invalid frozen IDs')
        lists[key]=ids
    saved=json.loads(source_config.read_text())
    values={**saved,**SLConfig.fromfile(str(ROOT/'configs/dome_transfer/d0_3e.py'))._cfg_dict.to_dict()}
    validate_config(values)
    values.update(eval=True,eval_ema=False,distributed=False,rank=0,world_size=1,
        coco_path=cli.data_root,resume='',pretrain_model_path=cli.pretrained,output_dir=str(out),
        device='cuda',amp=True,num_workers=0,batch_size=1,max_train_steps=0,max_eval_steps=0,
        stop_after_epochs=0,eval_query_floor=0)
    args=argparse.Namespace(**values)
    save(out/'resolved_config.json',values)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model,_,post=build_model_main(args);initialize(model,args);model.cuda().eval()
    report=dict(status='RUNNING',hashes=hashes,source_sha256=source_hashes(),command=sys.argv,
        gpu_environment=os.environ.get('CUDA_VISIBLE_DEVICES'),results={},
        limitation='Test engineering subset; no training. Proposal coverage is class-agnostic, not official recall.')
    save(out/'manifest.json',report)
    for phase,ids in lists.items():
        phase_dir=out/phase;phase_dir.mkdir();save(phase_dir/'ids.json',ids)
        args.eval_image_ids=str(phase_dir/'ids.json')
        _,dataset,_,loader,_=build_data_loaders(args)
        for group,mode in [('d0','spatial'),('d1','protected_density')]:
            model.transformer.proposal_selection_mode=mode
            folder=phase_dir/group;folder.mkdir()
            coco=get_coco_api_from_dataset(dataset)
            evaluators={k:CocoEvaluator(coco,['bbox']) for k in ('coarse','refined')}
            params=evaluators['refined'].coco_eval['bbox'].params
            vt_index=next((i for i,label in enumerate(params.areaRngLbl) if label in ('verytiny','very_tiny','very tiny')),None)
            if vt_index is None:raise ValueError('Evaluator lacks a recognized very-tiny area range')
            vt_low,vt_high=params.areaRng[vt_index]
            totals=dict(gt=0,hits025=0,hits050=0,vt=0,vt_hits050=0,queries=0)
            seen=[];errors={str(amp):0. for amp in (False,True)}
            with (folder/'diagnostics.jsonl').open('x') as stream,torch.no_grad():
                for samples,targets in loader:
                    image_id=int(targets[0]['image_id']);samples=samples.to('cuda')
                    # FP32 and AMP independent no-side-effect checks on the fixed32.
                    if phase=='precheck32':
                        for amp in (False,True):
                            with torch.autocast('cuda',enabled=amp):first=model(samples)
                            before={k:first[k].clone() for k in ('pred_logits','pred_boxes','pred_boxes_coarse','query_valid_mask')}
                            proposal_record(first,images[image_id],gt[image_id])
                            for k,v in before.items():assert torch.equal(v,first[k]),'Observer modified output'
                            with torch.autocast('cuda',enabled=amp):second=model(samples)
                            for k in ('pred_logits','pred_boxes','pred_boxes_coarse'):
                                errors[str(amp)]=max(errors[str(amp)],float((first[k].float()-second[k].float()).abs().max()))
                                torch.testing.assert_close(first[k].float(),second[k].float(),atol=1e-5,rtol=1e-4)
                        outputs=second
                    else:
                        with torch.autocast('cuda',enabled=True):outputs=model(samples)
                    sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
                    coarse,refined=paired_results(outputs,sizes,post['bbox'])
                    row=proposal_record(outputs,images[image_id],gt[image_id])
                    row['selection']=model.transformer.spatial_diagnostics
                    valid=outputs['query_valid_mask'][0]
                    row['coarse_cxcywh']=outputs['pred_boxes_coarse'][0,valid].float().cpu().tolist()
                    row['refined_cxcywh']=outputs['pred_boxes'][0,valid].float().cpu().tolist()
                    allocator=outputs['allocator_outputs'];p=allocator['density_prior']
                    y,mask=QueryBudgetLoss.build_density_targets(targets,p.shape[-2:],p.device,True,
                        spatial_valid_mask=allocator['density_valid_mask'],backend='vectorized')
                    under,support=underestimate_loss(p,y,mask)
                    row['density_under_raw']=float(under);row['density_support']=float(support)
                    centers=(y==1)&mask
                    row['density_center_mean']=float(p[centers].float().mean()) if centers.any() else None
                    row['predicted_count']=finite_json(allocator['predicted_count'])
                    row['predictions']={}
                    for k,result in [('coarse',coarse),('refined',refined)]:
                        pred={image_id:result[0]};row['predictions'][k]=evaluators[k].prepare(pred,'bbox')
                        evaluators[k].update(pred)
                    stream.write(json.dumps(finite_json(row),allow_nan=False)+'\n')
                    records=row['ground_truth'];vt=[g for g in records if vt_low<=g['area']<=vt_high]
                    totals['gt']+=len(records);totals['vt']+=len(vt)
                    totals['hits025']+=sum(g['best_proposal_iou']>=.25 for g in records)
                    totals['hits050']+=sum(g['best_proposal_iou']>=.5 for g in records)
                    totals['vt_hits050']+=sum(g['best_proposal_iou']>=.5 for g in vt)
                    totals['queries']+=row['executed_query_count'];seen.append(image_id)
                    if len(seen)%100==0:print(phase,group,len(seen),flush=True)
            if seen!=ids:raise ValueError('Image coverage/order mismatch')
            stats={}
            for key,evaluator in evaluators.items():
                evaluator.synchronize_between_processes();evaluator.accumulate();evaluator.summarize()
                stats[key]=evaluator.coco_eval['bbox'].stats.tolist()
                export_class_metrics(evaluator.coco_eval['bbox'],folder/(key+'_classes.json'))
                torch.save(evaluator.coco_eval['bbox'].eval,folder/(key+'_eval.pth'))
            if any(not np.isfinite(stats['refined'][i]) or stats['refined'][i]<0 for i in (0,4)):
                raise ValueError('AP/APvt has no valid support; cannot pass risk gate')
            report['results'][phase+'_'+group]=dict(totals=totals,stats=stats,export_max_error=errors,
                AP=stats['refined'][0]*100,APvt=stats['refined'][4]*100,
                vt_proposal050=100*totals['vt_hits050']/totals['vt'] if totals['vt'] else None)
            save(out/'report.json',report)
    a,b=(report['results']['subset1000_'+k] for k in ('d0','d1'))
    if a['vt_proposal050'] is None or b['vt_proposal050'] is None:raise ValueError('No VT support')
    checks=dict(AP=b['AP']>=a['AP']-1,APvt=b['APvt']>=a['APvt']-1,
        vt_proposal050=b['vt_proposal050']>=a['vt_proposal050']-2)
    if hashes!={str(p):sha256(p) for p in inputs}:raise ValueError('Inputs changed')
    report.update(status='COMPLETED',checks=checks,passed=all(checks.values()),inputs_unchanged=True)
    save(out/'report.json',report)
    return 0 if report['passed'] else 2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pretrained','data-root','precheck-ids','image-ids','output-dir'):p.add_argument('--'+key,required=True)
    cli=p.parse_args();out=Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False);print('Output:',out,flush=True)
    with (out/'console.log').open('x',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:
            code=run(cli,out)
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            save(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False))
            raise
    print('Completed; see report.json. Exit:',code,flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
