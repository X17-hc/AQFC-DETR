"""Manual, evaluation-only coarse/refined comparison from the SAME forward."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def paired_results(outputs, sizes, postprocess):
    import torch
    valid = outputs['query_valid_mask']
    if not torch.equal(valid.sum(1), outputs['executed_query_counts']):
        raise ValueError('Query mask/count mismatch')
    for key in ('pred_boxes', 'pred_boxes_coarse', 'pred_logits'):
        if not torch.isfinite(outputs[key][valid]).all():
            raise ValueError('Nonfinite valid prediction: ' + key)
    coarse = dict(outputs, pred_boxes=outputs['pred_boxes_coarse'])
    a, b = postprocess(coarse, sizes), postprocess(outputs, sizes)
    for x, y in zip(a, b):
        if not torch.equal(x['labels'], y['labels']) or not torch.equal(x['scores'], y['scores']):
            raise ValueError('Postprocessing changed labels/scores/order between branches')
    return a, b


def run(cli, out):
    import numpy as np
    import torch
    from main import build_model_main, build_data_loaders
    from datasets import get_coco_api_from_dataset
    from datasets.coco_eval import CocoEvaluator
    from util.checkpoint import load_native_resume
    from util.experiment import sha256, variant_signature
    from util.precision24 import is_fixed_six_refiner
    from util.config_validation import validate_config
    config_path = Path(cli.checkpoint).parent / 'config_args_all.json'
    annotation = Path(cli.data_root) / 'annotations/aitodv2_test.json'
    inputs = [Path(cli.checkpoint), config_path, annotation, Path(cli.precheck_ids), Path(cli.image_ids)]
    hashes = {str(p): sha256(p) for p in inputs}
    values = json.loads(config_path.read_text())
    validate_config(values)
    if not is_fixed_six_refiner(values):
        raise ValueError('Requires F2 fixed-six configuration')
    cp = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    if cp['epoch'] != 2 or not cp.get('run_metadata', {}).get('epoch_complete'):
        raise ValueError('Requires complete epoch2 checkpoint')
    if cp.get('variant_signature') != variant_signature(values):
        raise ValueError('Checkpoint and saved configuration signatures differ')
    del cp
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    values.update(eval=True, eval_ema=False, distributed=False, rank=0, world_size=1,
                  coco_path=cli.data_root, resume=cli.checkpoint, pretrain_model_path='',
                  output_dir=str(out), device='cuda', amp=True, num_workers=0,
                  max_train_steps=0, max_eval_steps=0, stop_after_epochs=0, eval_query_floor=0)
    args = argparse.Namespace(**values)
    known = {i['id'] for i in json.loads(annotation.read_text())['images']}
    lists = {}
    for name, path, count in [('precheck32', cli.precheck_ids, 32), ('subset1000', cli.image_ids, 1000)]:
        ids = json.loads(Path(path).read_text())
        if len(ids) != count or len(set(ids)) != count or any(type(i) is not int or i not in known for i in ids):
            raise ValueError('Invalid frozen IDs: ' + name)
        lists[name] = ids
    model, _, processors = build_model_main(args)
    model.to('cuda')
    load_native_resume(model, cli.checkpoint, expected_args=args)
    model.eval()
    report = dict(status='RUNNING', checkpoint=cli.checkpoint, hashes=hashes,
                  command=sys.argv, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  limitation='Test subset engineering diagnosis, not full-test AP or causal training ablation.',
                  branches='Same logits/query mask/top-k; only pred_boxes replaced by pred_boxes_coarse.',
                  results={})
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    save()
    try:
        for phase, ids in lists.items():
            folder = out/phase; folder.mkdir()
            id_path = folder/'ids.json'; id_path.write_text(json.dumps(ids))
            args.eval_image_ids = str(id_path)
            _, dataset, _, loader, _ = build_data_loaders(args)
            coco = get_coco_api_from_dataset(dataset)
            evaluators = {key: CocoEvaluator(coco, ['bbox']) for key in ('coarse','refined')}
            seen = []
            with contextlib.ExitStack() as stack, torch.no_grad():
                streams = {key: stack.enter_context((folder/(key+'_predictions.jsonl')).open('x')) for key in evaluators}
                raw = stack.enter_context((folder/'paired_queries.jsonl').open('x'))
                for samples, targets in loader:
                    samples = samples.to('cuda')
                    with torch.autocast('cuda', enabled=True):
                        outputs = model(samples)  # GT is never supplied to model decisions.
                    sizes = torch.stack([t['orig_size'] for t in targets]).to('cuda')
                    a, b = paired_results(outputs, sizes, processors['bbox'])
                    image_id = int(targets[0]['image_id']); seen.append(image_id)
                    valid = outputs['query_valid_mask'][0]
                    record = dict(image_id=image_id, original_size=sizes[0].tolist(),
                                  coordinate_format='normalized_cxcywh',
                                  query_indices=valid.nonzero().flatten().tolist())
                    for key in ('pred_boxes_coarse','pred_boxes','pred_logits'):
                        record[key] = outputs[key][0][valid].detach().float().cpu().tolist()
                    raw.write(json.dumps(record, allow_nan=False)+'\n')
                    for key, results in [('coarse', a), ('refined', b)]:
                        pred = {image_id: results[0]}
                        streams[key].write(json.dumps(dict(image_id=image_id, predictions=evaluators[key].prepare(pred,'bbox')), allow_nan=False)+'\n')
                        evaluators[key].update(pred)
                    if len(seen) % 100 == 0: print(phase, len(seen), '/',len(ids), flush=True)
            if seen != list(dataset.ids) or set(seen) != set(ids) or len(seen) != len(ids):
                raise ValueError('Image coverage/order failure')
            metrics = {}
            for key, evaluator in evaluators.items():
                print(phase, key, flush=True)
                evaluator.synchronize_between_processes(); evaluator.accumulate(); evaluator.summarize()
                metrics[key] = evaluator.coco_eval['bbox'].stats.tolist()
                torch.save(evaluator.coco_eval['bbox'].eval, folder/(key+'_eval.pth'))
            report['results'][phase] = dict(images=len(seen), metrics=metrics,
                delta_refined_minus_coarse_pp=[100*(b-a) for a,b in zip(metrics['coarse'],metrics['refined'])])
            save()
        if hashes != {str(p): sha256(p) for p in inputs}: raise ValueError('Inputs changed')
        report.update(status='COMPLETED', inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='FAILED',error=str(exc)); raise
    finally: save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint','data-root','precheck-ids','image-ids','output-dir'):
        p.add_argument('--'+flag, required=True)
    cli = p.parse_args()
    out = Path(cli.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    print('Same-forward coarse/refined; no training. Full log:',out/'console.log',flush=True)
    with (out/'console.log').open('x', encoding='utf-8', buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            run(cli,out)
    print('COMPLETED; see report.json',flush=True)


if __name__ == '__main__': main()
