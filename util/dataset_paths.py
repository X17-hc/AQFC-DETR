"""Dataset-specific paths shared by loaders and provenance records."""
from pathlib import Path


def visdrone_paths(root, split):
    root = Path(root)
    if split == 'eval_debug':
        split = 'val'
    if split not in ('train', 'val'):
        raise ValueError('VisDrone supports train/val only; do not silently map test or trainval to val')
    return (root / f'VisDrone2019-DET-{split}' / 'images',
            root / 'annotations_coco' / f'VisDrone2019-DET_{split}_coco.json')


def annotation_path(root, dataset, split):
    if dataset == 'visdrone':
        return visdrone_paths(root, split)[1]
    split = 'val' if split == 'eval_debug' else split
    if dataset == 'aitodv2':
        return Path(root) / 'annotations' / f'aitodv2_{split}.json'
    raise ValueError(f'Annotation provenance is not implemented for {dataset!r}')
