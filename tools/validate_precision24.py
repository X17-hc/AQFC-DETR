"""Synthetic validation only. No real dataset, trained checkpoint or download."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'outputs' / 'precision24' / 'synthetic' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', 'pytest', 'tests/test_precision24.py',
               '-q', '-s', '-ra', f'--junitxml={target / "tests.xml"}']
    print(f'Synthetic validation output: {target}', flush=True)
    with (target / 'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        code = process.wait()
    (target / 'result.json').write_text(json.dumps(dict(command=command, exit_code=code,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        limitation='Synthetic only. Read skipped cases in tests.xml; no AP/speed or full-training claim.'),
        indent=2), encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__': main()
