"""Manual F2 launcher: unique output, complete stdout/stderr, no task chaining."""
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


def launch():
    from main import get_args_parser, resolve_launch_defaults
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    from util.precision24 import is_fixed_six_refiner
    from util.epoch_boundary import validate_stop
    from util.experiment import unique_output, sha256
    parser = argparse.ArgumentParser(description=__doc__, parents=[get_args_parser()])
    args = resolve_launch_defaults(parser.parse_args())
    validate_stop(args)
    config = SLConfig.fromfile(args.config_file)
    if args.options: config.merge_from_dict(args.options)
    values = config._cfg_dict.to_dict()
    validate_config(values)
    if not is_fixed_six_refiner(values) or args.eval or args.p2_adapter or args.p2_acceptance:
        raise ValueError('This launcher is for F2 training only, without P2 transfer')
    if not args.resume and values['epochs'] == 24 and args.stop_after_epochs != 1:
        raise ValueError('A fresh F2 24-epoch recipe must stop after its first full epoch for review')
    if args.resume:
        import torch
        cp = torch.load(args.resume, map_location='cpu', weights_only=False)
        if cp.get('evaluation_state') == 'pending':
            raise ValueError('Training is saved but evaluation is pending. Review/recover evaluation first; no automatic continuation.')
        path = Path(args.output_dir)/'first_epoch_review.json'
        review = json.loads(path.read_text()) if path.is_file() else {}
        if review.get('status') != 'READY_FOR_MANUAL_REVIEW':
            raise ValueError('F2 continuation requires the completed first-epoch review in the original output directory')
        if Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
            raise ValueError('Resume must use its original experiment output directory')
        if cp.get('epoch') == 0 and sha256(args.resume) != review.get('checkpoint_sha256'):
            raise ValueError('Resume checkpoint does not match first-epoch review')
        del cp
    unique_output(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    launch_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    log_dir = out/'launches'/launch_id
    log_dir.mkdir(parents=True, exist_ok=False)
    # Retain provenance that the main entry refreshes during an exact resume.
    if args.resume:
        import shutil
        for name in ['experiment_manifest.json','config_args_all.json','config_args_raw.json','config_cfg.py']:
            if (out/name).is_file(): shutil.copy2(out/name,log_dir/('previous_'+name))
    argv = sys.argv[1:]
    if '--unique-output-dir' in argv: argv.remove('--unique-output-dir')
    if '--output-dir' in argv:
        argv[argv.index('--output-dir')+1] = str(out.resolve())
    else:
        argv += ['--output-dir',str(out.resolve())]
    command = [sys.executable,'-u',str(ROOT/'main.py'),*argv]
    record = dict(command=command, output_dir=str(out), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  config_sha256=sha256(args.config_file), status='RUNNING', automatic_continuation=False)
    (log_dir/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Output:',out,'\nComplete console:',log_dir/'console.log',flush=True)
    with (log_dir/'console.log').open('x',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            # Only this launcher's child; preserve epoch-boundary recovery files.
            import signal
            process.send_signal(signal.SIGINT)
            code = process.wait()
        finally:
            record.update(exit_code=process.poll(),status='COMPLETED' if process.poll()==0 else 'FAILED_OR_INTERRUPTED')
            (log_dir/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__': launch()
