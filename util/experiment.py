"""Run provenance and explicit experiment compatibility; no model dependencies."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


UPDATE_DEFAULTS = dict(allocator_encoder_type='standard', allocator_enabled=True,
                       calibrator_enabled=True, density_target_backend='reference',
                       density_target_chunk_size=512, spatial_semantic_ratio=.75,
                       spatial_grid_size=[8, 8], eval_backend='legacy',
                       run_purpose='engineering_check')


def variant_signature(args):
    values = vars(args) if not isinstance(args, dict) else args
    keys = ('query_budget_levels', 'force_query_budget', 'proposal_selection_mode',
            'proposal_density_weight', 'mixed_density_ratio', 'train_split', 'eval_split',
            'calibrator_gate_type', 'calibrator_use_spatial', 'calibrator_spatial_alphas',
            'allocator_schedule', 'grouped_decoder_inference')
    result = {k: values.get(k, v) for k, v in UPDATE_DEFAULTS.items()
              if k not in ('run_purpose', 'eval_backend')}
    result.update({k: values.get(k) for k in keys})
    if values.get('dataset_file') == 'visdrone':
        result['dataset_contract'] = 'visdrone_raw_ids_1_10_coco_bbox_proxy'
    result.update({k: v for k, v in values.items()
                   if k.endswith(('_loss_weight', '_loss_coef')) or k in
                   ('lr', 'lr_backbone', 'weight_decay', 'use_dn', 'dn_number',
                    'dn_box_noise_scale', 'dn_label_noise_ratio', 'use_ema', 'amp')})
    return json.loads(json.dumps(result, default=str))


def original_variant(args):
    sig = variant_signature(args)
    return (sig['allocator_encoder_type'] == 'standard' and sig['allocator_enabled']
            and sig['calibrator_enabled'] and sig['density_target_backend'] == 'reference'
            and sig['proposal_selection_mode'] != 'spatial')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unique_output(args):
    if not getattr(args, 'unique_output_dir', False):
        return
    if args.resume:
        raise ValueError('--unique-output-dir cannot be used with --resume')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = root / stamp
    path.mkdir(exist_ok=False)
    args.output_dir = str(path.resolve())


def write_manifest(args, project_root, update=None):
    import torch
    path = Path(args.output_dir) / 'experiment_manifest.json'
    if update is not None:
        data = json.loads(path.read_text(encoding='utf-8'))
        data.update(update)
    else:
        def git(*command):
            try:
                return subprocess.check_output(['git', *command], cwd=project_root,
                                               stderr=subprocess.DEVNULL, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
        annotations = []
        for split in sorted({args.train_split, args.eval_split}):
            from .dataset_paths import annotation_path
            try:
                source = annotation_path(args.coco_path, getattr(args, 'dataset_file', 'aitodv2'), split)
            except ValueError:
                annotations.append(dict(split=split, status='annotation_path_unresolved'))
                continue
            if source.is_file():
                raw = json.loads(source.read_text(encoding='utf-8'))
                annotations.append(dict(path=str(source), sha256=sha256(source), split=split,
                                        categories=raw.get('categories', [])))
            else:
                annotations.append(dict(split=split, status='annotation_path_unresolved'))
        weight = args.resume or args.pretrain_model_path
        report_path = Path(args.output_dir) / 'checkpoint_migration_report.json'
        data = dict(created_at=datetime.now().astimezone().isoformat(),
                    command=sys.argv, config={**UPDATE_DEFAULTS, **vars(args)}, signature=variant_signature(args),
                    run_purpose=getattr(args, 'run_purpose', 'engineering_check'),
                    eval_backend=getattr(args, 'eval_backend', 'legacy'), dataset_root=str(args.coco_path),
                    git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                    annotations=annotations, initialization=dict(path=weight or None,
                        sha256=sha256(weight) if weight and Path(weight).is_file() else None,
                        training_provenance='unknown' if weight else 'ImageNet backbone only'),
                    migration_report=json.loads(report_path.read_text(encoding='utf-8'))
                        if report_path.exists() else None,
                    environment=dict(python=sys.version, torch=torch.__version__,
                        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()
                        if torch.cuda.is_available() else None), epoch_complete=False)
        data['git_dirty'] = bool(data['git_status']) if data['git_status'] is not None else None
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
