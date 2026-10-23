"""Verify the published historical snapshot without importing training code."""
import ast
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    evidence = root / 'evidence/pretrain24_20261015'
    manifest = json.loads((evidence / 'published_source_manifest.json').read_text(encoding='utf8'))
    assert len(manifest) == 175
    assert len({entry['path'] for entry in manifest}) == len(manifest)
    syntax_count = 0
    base_count = 0
    for entry in manifest:
        file = root / entry['path']
        data = file.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
        if file.suffix == '.py':
            tree = ast.parse(data, filename=entry['path'])
            syntax_count += 1
            if entry['path'].startswith('configs/'):
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
                        bases = ast.literal_eval(node.value)
                        for base in ([bases] if isinstance(bases, str) else bases):
                            assert (file.parent / base).is_file(), (entry['path'], base)
                            base_count += 1
    rows = [json.loads(line) for line in (evidence / 'run_pretrain24/metrics.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    assert len(rows) == 24 and [row['epoch'] for row in rows] == list(range(24))
    final = rows[-1]
    metrics = final['test_bbox_metrics']
    assert abs(metrics['AP'] - 0.3196184653832317) < 1e-14
    assert abs(metrics['APvt'] - 0.16555134145271416) < 1e-14
    assert final['test_evaluated_images'] == 14018
    run = json.loads((evidence / 'run_pretrain24/experiment_manifest.json').read_text(encoding='utf8'))
    config = run['config']
    assert config['epochs'] == 24 and config['seed'] == 42 and config['batch_size'] == 2
    assert config['eval_backend'] == 'legacy' and config['eval_split'] == 'test'
    assert config['train_split'] == 'trainval' and config['use_ema'] is False
    assert run['initialization']['sha256'] == 'e6c3723b0f82c7aa6a6954576ad4e572bab1f87b57281069dd1814afedc16abe'
    status = json.loads((evidence / 'status.json').read_text(encoding='utf8'))
    assert status['status'] == 'complete' and status['last_epoch'] == 23
    subset = json.loads((evidence / 'subset_provenance.json').read_text(encoding='utf8'))
    assert hashlib.sha256((evidence / 'subset_ids.json').read_bytes()).hexdigest() == subset['sha256']
    for entry in json.loads((evidence / 'evaluator_source_verification.json').read_text(encoding='utf8')):
        data = (root / entry['path']).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == entry['server_installed_lf_sha256']
    return {'source_hashes_verified': len(manifest), 'python_files_parsed': syntax_count,
            'config_base_references_verified': base_count, 'epoch_records_verified': len(rows),
            'AP': metrics['AP'] * 100, 'APvt': metrics['APvt'] * 100,
            'training_or_inference_executed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
