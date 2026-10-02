"""Read-only H1/D0/D1 class and area slices. Never writes checkpoints."""
import argparse
import json
from pathlib import Path

H1_CLASSES = (
    'airplane', 'bridge', 'storage-tank', 'ship',
    'swimming-pool', 'vehicle', 'person', 'wind-mill',
)
H1_FULL = dict(AP=0.322, AP50=0.682, AP75=0.263, APvt=0.156)
D0_FULL = dict(AP=0.322, AP50=0.677, AP75=0.265, APvt=0.160, APs=0.372)
D1_FULL = dict(AP=0.324, AP50=0.678, AP75=0.268, APvt=0.161, APs=0.377)


def load_class_metrics(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    classes = payload.get('classes')
    if not isinstance(classes, list) or not classes:
        raise ValueError('class_metrics.json missing classes')
    return payload


def summarize(payload):
    rows = []
    for item in payload['classes']:
        areas = item.get('areas') or {}
        rows.append(dict(
            category_id=item.get('category_id'),
            name=item.get('name'),
            AP=(areas.get('all') or {}).get('AP'),
            APvt=(areas.get('verytiny') or areas.get('very tiny') or {}).get('AP'),
            AR=(areas.get('all') or {}).get('AR'),
        ))
    finite_ap = [r['AP'] for r in rows if isinstance(r['AP'], (int, float))]
    finite_vt = [r['APvt'] for r in rows if isinstance(r['APvt'], (int, float))]
    return dict(
        classes=rows,
        mean_AP=sum(finite_ap) / len(finite_ap) if finite_ap else None,
        mean_APvt=sum(finite_vt) / len(finite_vt) if finite_vt else None,
        vehicle_person=[r for r in rows if r['name'] in ('vehicle', 'person')],
    )


def compare_runs(baseline, candidate):
    left = {r['name']: r for r in summarize(baseline)['classes']}
    right = {r['name']: r for r in summarize(candidate)['classes']}
    rows = []
    for name in left.keys() | right.keys():
        a, b = left.get(name, {}), right.get(name, {})
        rows.append(dict(
            name=name,
            dAP=_sub(b.get('AP'), a.get('AP')),
            dAPvt=_sub(b.get('APvt'), a.get('APvt')),
        ))
    return dict(classes=rows)


def _sub(new, old):
    if new is None or old is None:
        return None
    return float(new) - float(old)


def hypotheses_from_evidence(metrics):
    h1, d0, d1 = metrics['h1_full'], metrics['d0_full'], metrics['d1_full']
    return [
        dict(
            id='H-cls',
            claim='H1 相对论文 DQ 主要赢在 AP75、输在 AP50，下一刀应动分类/匹配而不是再加框损失。',
            evidence=f"H1 AP50={h1['AP50']} AP75={h1['AP75']} vs DQ 0.692/0.227",
            probe='h1_ema_3e 或 joint o2m 下调',
            falsify='全量 AP50 再降且 AP75 不升，或 EMA/o2m 探针 AP75≤H1-0.2pp',
        ),
        dict(
            id='H-vt',
            claim='D1 组合抬了 APs，几乎没抬 APvt，密度包没有打到 verytiny。',
            evidence=f"D1-D0 APvt={d1['APvt']-d0['APvt']:+.3f} APs={d1.get('APs',0)-d0.get('APs',0):+.3f}",
            probe='Dome 思路拆开：只开 protected_density 或只开低估项',
            falsify='拆开后全量 APvt 相对 D0/H1 ≥+0.3pp，或 class_metrics 里 vehicle/person VT 先动',
        ),
        dict(
            id='H-ema-joint',
            claim='H1 关 EMA、joint DFL/o2m 未单拆；3e 低 LR 微调足以证伪权重/平滑假设。',
            evidence=f"H1 AP={h1['AP']} APvt={h1['APvt']}；D0 同 AP、APvt+0.4pp",
            probe='h1_ema_3e、h1_joint_dfl_035_3e、h1_glu_3e 互斥',
            falsify='任一条未过 AP+0.3 且 APvt+0.3 且 AP75≥-0.2pp 则该假设灭',
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-metrics', action='append', default=[],
                        help='Repeatable class_metrics.json paths, name=path')
    parser.add_argument('--output', default='', help='Optional JSON write path')
    args = parser.parse_args()
    runs = {}
    for item in args.class_metrics:
        name, _, path = item.partition('=')
        if not path:
            name, path = Path(item).stem, item
        runs[name] = summarize(load_class_metrics(path))
    report = dict(
        runs=runs,
        published=dict(h1=H1_FULL, d0=D0_FULL, d1=D1_FULL),
        hypotheses=hypotheses_from_evidence(dict(h1_full=H1_FULL, d0_full=D0_FULL, d1_full=D1_FULL)),
        note='Console 没有八类数组时只用 published 全量数字；有 class_metrics 才填 runs。',
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
