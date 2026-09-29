"""Fixed-budget parallel redundancy priority; not greedy NMS and never GT-driven."""
import math
import torch
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from .spatial_selection import spatial_indices, capped_quotas, _quota_additions

@torch.no_grad()
def redundancy_mask(boxes, labels, density, valid_classes, chunk=256):
    """Inputs sorted by semantic score/token id; every higher-ranked pool box counts."""
    if chunk < 1:
        raise ValueError('chunk must be positive')
    xyxy = box_cxcywh_to_xyxy(boxes.float())
    legal = torch.zeros_like(labels, dtype=torch.bool)
    for category in valid_classes:
        legal |= labels == category
    columns = torch.arange(len(boxes), device=boxes.device)
    pieces = []
    for start in range(0, len(boxes), chunk):
        end = min(start+chunk, len(boxes))
        iou = box_iou(xyxy[start:end], xyxy)[0]
        threshold = .4 + .5 * torch.maximum(density[start:end,None].float(), density[None].float())
        higher = columns[None] < columns[start:end,None]
        same = labels[start:end,None] == labels[None]
        pieces.append(((iou > threshold) & higher & same & legal[start:end,None] & legal[None]).any(1))
    return torch.cat(pieces) if pieces else legal

def token_grid(padding, density, shapes, grid):
    gh, gw = grid
    cells, levels, start = [], [], 0
    weights = density.new_zeros(gh*gw, dtype=torch.float32)
    for level, (h,w) in enumerate(shapes):
        h,w = int(h),int(w)
        content = ~padding[start:start+h*w].reshape(h,w)
        vh, vw = max(1,int(content.any(1).sum())), max(1,int(content.any(0).sum()))
        yy,xx = torch.meshgrid(torch.arange(h,device=density.device),torch.arange(w,device=density.device),indexing='ij')
        cell = (((yy+.5)/vh*gh).long().clamp(0,gh-1)*gw + ((xx+.5)/vw*gw).long().clamp(0,gw-1)).flatten()
        cells.append(cell); levels.append(torch.full_like(cell,level))
        if level==0:
            support=content.flatten()
            values=density[:h*w].float()
            sums=torch.zeros_like(weights).scatter_add_(0,cell[support],values[support])
            sizes=torch.zeros_like(weights).scatter_add_(0,cell[support],torch.ones_like(values[support]))
            weights=sums/sizes.clamp_min(1)
        start+=h*w
    return torch.cat(cells),torch.cat(levels),weights

@torch.no_grad()
def protected_indices(logits, density, padding, invalid, topk, shapes, decoded_boxes,
                      density_weight=.25, ratio=.75, grid=(8,8), diagnostics=None,
                      valid_classes=tuple(range(8))):
    if decoded_boxes is None or decoded_boxes.shape != (*logits.shape[:2],4):
        raise ValueError('protected_density requires independent decoded_proposal_boxes [B,S,4]')
    if shapes is None:
        raise ValueError('protected_density requires spatial_shapes')
    shapes=shapes.tolist() if isinstance(shapes,torch.Tensor) else shapes
    if sum(int(h)*int(w) for h,w in shapes)!=logits.shape[1]:
        raise ValueError('spatial_shapes do not match tokens')
    semantic,labels=logits.float().max(-1)
    rows=[]
    for b in range(len(logits)):
        valid=~invalid[b]
        if (not torch.isfinite(logits[b,valid]).all() or not torch.isfinite(density[b,~padding[b]]).all()
                or not torch.isfinite(decoded_boxes[b,valid]).all()):
            raise FloatingPointError('Nonfinite protected-density input at valid positions')
        if ((density[b,~padding[b]]<0)|(density[b,~padding[b]]>1)).any():
            raise ValueError('density probabilities must be in [0,1]')
        if (decoded_boxes[b,valid,2:]<=0).any():
            raise ValueError('decoded proposal widths/heights must be positive')
        n=int(valid.sum()); count=min(int(topk),n)
        if count<1:
            raise ValueError('Each image needs at least one valid proposal')
        order=torch.argsort(semantic[b].masked_fill(~valid,-torch.inf),descending=True,stable=True)[:n]
        cells,levels,weights=token_grid(padding[b],density[b],shapes,grid)
        reserve=math.ceil(ratio*count)
        core=order[:reserve]
        fallback=not bool((density[b,~padding[b]]>0).any())
        good=bad=filled=0
        if fallback:
            chosen=order[:count]
        else:
            original=spatial_indices(semantic[b:b+1],density[b:b+1],padding[b:b+1],invalid[b:b+1],
                count,shapes,density_weight,ratio,grid)[0]
            in_pool=torch.zeros_like(valid)
            in_pool[original]=True
            in_pool[order[:min(n,max(1500,2*count))]]=True
            pool=order[in_pool[order]]
            redundant=redundancy_mask(decoded_boxes[b,pool],labels[b,pool],density[b,pool],valid_classes)
            is_redundant=torch.zeros_like(valid); is_redundant[pool]=redundant
            remaining=in_pool & valid; remaining[core]=False
            capacities=torch.bincount(cells[remaining],minlength=grid[0]*grid[1]).tolist()
            quotas=capped_quotas(weights.tolist(),capacities,count-reserve)
            # Stable priority sort keeps semantic/token ordering within each tier.
            prioritized=pool[torch.argsort(redundant.long(),stable=True)]
            additions=_quota_additions(prioritized,remaining,cells,quotas)
            bad=int(is_redundant[additions].sum()); good=len(additions)-bad
            chosen=torch.cat([core,additions])
            if len(chosen)<count:
                left=valid.clone(); left[chosen]=False
                extra=order[left[order]][:count-len(chosen)]
                filled=len(extra); chosen=torch.cat([chosen,extra])
            # All gathers use this single ordering, including ties.
            chosen=chosen.sort().values
            chosen=chosen[torch.argsort(semantic[b,chosen],descending=True,stable=True)]
        if diagnostics is not None:
            diagnostics.append(dict(core_count=reserve, nonredundant_added=good,
                redundant_added=bad, global_filled=filled, fallback=int(fallback),
                semantic_fallback_count=count-reserve if fallback else 0,
                grid_coverage=cells[chosen].unique().numel()/(grid[0]*grid[1]),
                level_counts=torch.bincount(levels[chosen],minlength=len(shapes)).tolist()))
        rows.append(torch.cat([chosen,chosen[:1].expand(int(topk)-count)]))
    return torch.stack(rows)
