"""Manual H1 entry with complete console capture; never chains experiments."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from main import get_args_parser, resolve_launch_defaults
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    from util.epoch_boundary import validate_stop
    from util.experiment import unique_output, sha256
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = resolve_launch_defaults(parser.parse_args())
    validate_stop(args)
    config = SLConfig.fromfile(args.config_file)
    if args.options: config.merge_from_dict(args.options)
    v = config._cfg_dict.to_dict()
    validate_config(v)
    if v.get('architecture_variant') != 'legacy_joint_v2' or args.eval or args.p2_adapter or args.p2_acceptance:
        raise ValueError('H1 training launcher requires H1 without transfer/eval options')
    if args.eval_image_ids:
        raise ValueError('H1 full validation cannot be restricted by --eval-image-ids')
    if args.max_eval_steps and not args.max_train_steps:
        raise ValueError('Partial evaluation is allowed only in the explicit training smoke run')
    if not args.max_train_steps and not Path(v['joint_subset_ids']).is_file():
        raise FileNotFoundError('Fixed subset IDs are required before long training')
    subset_sha = None
    if not args.max_train_steps:
        raw = json.loads(Path(v['joint_subset_ids']).read_text(encoding='utf-8'))
        ids = raw if isinstance(raw,list) else raw.get('image_ids',raw.get('ids'))
        annotation = Path(args.coco_path)/'annotations/aitodv2_test.json'
        known = {im['id'] for im in json.loads(annotation.read_text(encoding='utf-8'))['images']}
        if not isinstance(ids,list) or len(ids)!=1000 or len(set(ids))!=1000 or set(ids)-known:
            raise ValueError('H1 requires 1000 unique known test image IDs before training')
        subset_sha = sha256(v['joint_subset_ids'])
    if args.resume:
        import torch
        cp = torch.load(args.resume, map_location='cpu', weights_only=False)
        if cp.get('evaluation_state') == 'pending':
            raise ValueError('Complete training saved, but evaluation is pending; review evaluation before continuation')
        if Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
            raise ValueError('Resume uses the original output directory')
        del cp
    unique_output(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(out.iterdir()):
        raise FileExistsError('Fresh H1 run requires an empty or unique output directory')
    logdir = out/'launches'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    logdir.mkdir(parents=True, exist_ok=False)
    if args.resume:
        import shutil
        for name in ('experiment_manifest.json','config_args_all.json','config_cfg.py'):
            if (out/name).is_file(): shutil.copy2(out/name, logdir/('previous_'+name))
    argv = list(sys.argv[1:])
    if '--unique-output-dir' in argv: argv.remove('--unique-output-dir')
    if '--output-dir' in argv: argv[argv.index('--output-dir')+1] = str(out.resolve())
    else: argv += ['--output-dir',str(out.resolve())]
    command = [sys.executable,'-u',str(ROOT/'main.py'),*argv]
    record = dict(command=command, config_sha256=sha256(args.config_file),
                  subset_sha256=subset_sha,
                  gpu_environment=os.environ.get('CUDA_VISIBLE_DEVICES'), status='RUNNING', automatic_continuation=False)
    from tools.benchmark_interleaved import source_hashes
    record['source_sha256'] = source_hashes()
    from tools.benchmark_control import snapshot
    record['resources_before'] = snapshot()  # Read-only, not an occupancy intercept.
    (logdir/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Output:',out,'\nComplete console:',logdir/'console.log',flush=True)
    with (logdir/'console.log').open('x',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGINT); code = process.wait()
        finally:
            record.update(exit_code=process.poll(),status='COMPLETED' if process.poll()==0 else 'FAILED_OR_INTERRUPTED')
            (logdir/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__': main()
