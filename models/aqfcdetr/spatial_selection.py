"""Deterministic budget-constrained spatial coverage (no GT, no NMS)."""
import math
import torch


def capped_quotas(weights, capacities, total):
    """Largest remainder with redistribution after capacity saturation."""
    quotas = [0] * len(weights)
    while total:
        active = [i for i, cap in enumerate(capacities) if cap > quotas[i]]
        if not active:
            break
        denom = sum(weights[i] for i in active)
        shares = {i: total * (weights[i] / denom if denom > 0 else 1 / len(active))
                  for i in active}
        for i in active:
            take = min(capacities[i] - quotas[i], int(shares[i]))
            quotas[i] += take
            total -= take
        if not total:
            break
        for i in sorted(active, key=lambda j: (-(shares[j] % 1), j)):
            if quotas[i] < capacities[i] and total:
                quotas[i] += 1
                total -= 1
    return quotas


@torch.no_grad()
def spatial_indices(semantic, density, padding, invalid, topk, shapes,
                    density_weight=.25, ratio=.75, grid=(8, 8), diagnostics=None):
    if shapes is None:
        raise ValueError('Spatial selection requires spatial_shapes')
    shapes = shapes.tolist() if isinstance(shapes, torch.Tensor) else shapes
    if sum(int(h) * int(w) for h, w in shapes) != semantic.shape[1]:
        raise ValueError('spatial_shapes do not match encoder tokens')
    gh, gw = grid
    rows, reports = [], []
    for b in range(len(semantic)):
        valid = ~invalid[b]
        if not torch.isfinite(semantic[b, valid]).all():
            raise FloatingPointError('Nonfinite semantic logits in valid spatial candidates')
        count = min(topk, int(valid.sum()))
        if count < 1:
            raise ValueError('Each image needs at least one valid proposal')
        semantic_order = torch.argsort(semantic[b].masked_fill(~valid, -torch.inf),
                                       descending=True, stable=True)[:count]
        cells, levels, start = [], [], 0
        weights = torch.zeros(gh * gw, device=density.device)
        for level, (h, w) in enumerate(shapes):
            h, w = int(h), int(w)
            content = ~padding[b, start:start + h * w].reshape(h, w)
            vh, vw = max(1, int(content.any(1).sum())), max(1, int(content.any(0).sum()))
            yy, xx = torch.meshgrid(torch.arange(h, device=density.device),
                                    torch.arange(w, device=density.device), indexing='ij')
            cell = (((yy + .5) / vh * gh).long().clamp(0, gh - 1) * gw
                    + ((xx + .5) / vw * gw).long().clamp(0, gw - 1)).flatten()
            cells.append(cell)
            levels.append(torch.full_like(cell, level))
            if level == 0:
                values = density[b, :h * w]
                support = content.flatten()
                sums = torch.zeros_like(weights).scatter_add_(0, cell[support], values[support])
                sizes = torch.zeros_like(weights).scatter_add_(0, cell[support],
                                                               torch.ones_like(values[support]))
                weights = sums / sizes.clamp(min=1)
            start += h * w
        cells, levels = torch.cat(cells), torch.cat(levels)
        finite = bool(torch.isfinite(density[b, ~padding[b]]).all())
        joint = semantic[b] + density_weight * torch.logit(density[b].float().clamp(1e-4, 1-1e-4))
        reserve = min(count, math.ceil(ratio * count))
        chosen = semantic_order[:reserve]
        remaining = valid.clone()
        remaining[chosen] = False
        capacities = torch.bincount(cells[remaining], minlength=gh * gw).tolist()
        weight_values = weights.tolist()
        available_weight = sum(weight_values[i] for i, cap in enumerate(capacities) if cap)
        fallback = not finite or (count > reserve and available_weight <= 0)
        fill_count = 0
        if fallback:
            chosen = semantic_order
        else:
            quotas = capped_quotas(weight_values, capacities, count - reserve)
            ordered = torch.argsort(joint.masked_fill(~remaining, -torch.inf),
                                    descending=True, stable=True)
            additions = [ordered[remaining[ordered] & (cells[ordered] == cell)][:quota]
                         for cell, quota in enumerate(quotas) if quota]
            if additions:
                chosen = torch.cat([chosen] + additions)
            if len(chosen) < count:
                remaining[chosen] = False
                extra = ordered[remaining[ordered]][:count - len(chosen)]
                fill_count = len(extra)
                chosen = torch.cat([chosen, extra])
            # Sorting indices first supplies a global token-id tie breaker.
            chosen = chosen.sort().values
            chosen = chosen[torch.argsort(joint[chosen], descending=True, stable=True)]
        reports.append(dict(semantic_reserved=count if fallback else reserve,
                            spatial_added=0 if fallback else count-reserve-fill_count,
                            filled=fill_count, fallback=int(fallback),
                            grid_coverage=float(cells[chosen].unique().numel() / (gh*gw)),
                            level_counts=torch.bincount(levels[chosen], minlength=len(shapes)).tolist()))
        rows.append(torch.cat([chosen, chosen[:1].expand(topk-count)]))
    if diagnostics is not None:
        diagnostics.extend(reports)
    return torch.stack(rows)
