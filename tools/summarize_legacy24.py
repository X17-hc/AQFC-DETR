"""Join saved paired GT errors to encoder proposal coverage; no model execution."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--before',choices=['D0','D1'],default='D0')
    p.add_argument('--after',choices=['D1','D2'],default='D1')
    p.add_argument('--paired',type=Path,required=True)
    a=p.parse_args(); out=a.root
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from util.error_analysis import area_bucket,density_bucket
    source={}; metrics={}; queries={}
    for group in [a.before,a.after]:
        if read(out/group/'diagnosis_status.json')['status']!='completed': raise ValueError('Incomplete run')
        records=list(rows(out/group/'metrics.jsonl')); assert len(records)==1
        r=records[0]; assert r['test_evaluated_images']==1000
        metrics[group]=r['test_bbox_metrics'];queries[group]=dict(mean=r['test_mean_executed_queries'],total=r['test_decoder_query_tokens'])
        source[group]={}
        image_ids=[]
        for image in rows(out/group/'proposals.jsonl'):
            image_ids.append(image['image_id'])
            if len(image['proposals'])!=image['executed_query_count']: raise ValueError('Mask/count mismatch')
            for gt in image['ground_truth']:
                key=(image['image_id'],gt['annotation_id'])
                if key in source[group]: raise ValueError('Duplicate GT')
                source[group][key]=gt
        assert len(image_ids)==len(set(image_ids))==1000
    assert source[a.before].keys()==source[a.after].keys()
    for k in source[a.before]:
        assert {n:source[a.before][k][n] for n in ['bbox','category_id','area']}=={n:source[a.after][k][n] for n in ['bbox','category_id','area']}
    groups=defaultdict(lambda:{'gt':0,a.before:Counter(),a.after:Counter()})
    image_count=Counter(k[0] for k in source[a.before])
    support=Counter(); support_vt=Counter()
    jointpath=out/(a.before+'_'+a.after+'_proposal_head_gt.jsonl')
    with jointpath.open('x',encoding='utf-8') as stream:
        for row in rows(a.paired/'paired_gt.jsonl'):
            key=(row['image_id'],row['annotation_id']); support[row['category_id']]+=1
            if 0<=row['area']<=64: support_vt[row['category_id']]+=1
            coverage={g:source[g][key]['best_proposal_iou'] for g in [a.before,a.after]}
            stream.write(json.dumps({**row,'labels':{'epoch0':a.before,'epoch1':a.after},'proposal_best_iou':coverage},separators=(',',':'))+'\n')
            bins=['overall','class:'+str(row['category_id']),'size:'+area_bucket(row['area']),
                  'density:'+density_bucket(image_count[row['image_id']]),'original_K:'+str(row['original_K'])]
            for name in bins:
                g=groups[name];g['gt']+=1
                for group,label in [(a.before,'epoch0'),(a.after,'epoch1')]:
                    c=g[group]; prior=coverage[group]; f=row[label]
                    c['proposal_hit025']+=int(prior>=.25);c['proposal_hit050']+=int(prior>=.5)
                    for t in ['50','75']:
                        v=f['thresholds'][t]
                        c['final_same_hit'+t]+=int(v['covered'])
                        c['prior050_final_missing'+t]+=int(prior>=.5 and not v['covered'])
                        c['prior_below050_final_hit'+t]+=int(prior<.5 and v['covered'])
                        c['low_score'+t]+=int(v['low_score']);c['ranking'+t]+=int(v['ranking_inconsistent'])
                        c['low_and_ranking'+t]+=int(v['low_and_ranking'])
    summary={'labels':[a.before,a.after],'metrics':metrics,'queries':queries,'coverage':groups,
             'support':support,'support_vt_inclusive':support_vt,'scope':'subset descriptive; proposal many-to-one; no causal proof'}
    save(out/(a.before+'_'+a.after+'_joined_summary.json'),summary)
    if a.before=='D0':
        delta={k:(metrics['D1'][k]-metrics['D0'][k])*100 for k in ['AP','AP75','APvt']}
        supported={}
        for group in ['D0','D1']:
            e=torch.load(out/group/'eval.pth',map_location='cpu',weights_only=False)
            cids=e['params'].catIds
            supported[group]={}
            for key,area,s in [('AP',0,support),('APvt',1,support_vt)]:
                means=[]
                for ix,cid in enumerate(cids):
                    if s[int(cid)]>=100:
                        values=e['precision'][:,:,ix,area,-1]; values=values[values>=0]
                        if values.size: means.append(float(values.mean()))
                supported[group][key]=float(np.mean(means)) if means else None
        robust={k:100*(supported['D1'][k]-supported['D0'][k]) if supported['D1'][k] is not None else None for k in ['AP','APvt']}
        g=groups['overall']
        geometry=any(g['D1'][k]>g['D0'][k] for k in ['proposal_hit050','final_same_hit50','final_same_hit75'])
        signal=any(delta[k]>=.3 and robust[k] is not None and robust[k]>=.3 for k in ['AP','APvt']) and min(delta.values())>=-.2 and geometry
        missing=g['gt']-g['D1']['proposal_hit050']
        decision={'delta_pp':delta,'high_support_macro_delta_pp':robust,'geometry_improves':geometry,'budget_signal':signal,
                  'run_D2':not signal and missing>0,'D1_uncovered_proposal_gt':missing,
                  'rule':'AP/APvt +0.3pp with >=100-GT class-only macro also +0.3pp; all three >=-0.2pp; geometry direction positive',
                  'note':'conservative low-support safeguard; descriptive screening, not significance'}
        save(out/'decision.json',decision); print(json.dumps(decision,indent=2))

if __name__=='__main__': main()
