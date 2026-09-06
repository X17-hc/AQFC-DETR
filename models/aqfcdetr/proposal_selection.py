import math

import torch


def select_proposal_indices(class_logits, density_prior, padding_mask, topk,
                            mode='fused', density_weight=0.25,
                            mixed_density_ratio=0.25):
    """Select valid encoder-token indices with semantic/density ranking."""
    semantic_logits = class_logits.max(dim=-1).values
    density_logits = torch.logit(density_prior.clamp(1e-4, 1.0 - 1e-4))
    joint_scores = semantic_logits + density_weight * density_logits
    invalid = padding_mask.bool()
    semantic_logits = semantic_logits.masked_fill(invalid, float('-inf'))
    density_logits = density_logits.masked_fill(invalid, float('-inf'))
    joint_scores = joint_scores.masked_fill(invalid, float('-inf'))
    valid_minimum = int((~invalid).sum(dim=1).min().item())
    topk = min(int(topk), joint_scores.shape[1], valid_minimum)
    if topk <= 0:
        return torch.empty(class_logits.shape[0], 0, dtype=torch.long,
                           device=class_logits.device)

    if mode == 'semantic' or density_weight == 0.0 and mode == 'fused':
        return torch.topk(semantic_logits, topk, dim=1).indices
    if mode == 'fused':
        return torch.topk(joint_scores, topk, dim=1).indices
    if mode != 'mixed':
        raise ValueError(f'Unknown proposal selection mode: {mode}')

    semantic_k = int(math.ceil((1.0 - mixed_density_ratio) * topk))
    selections = []
    for batch_index in range(class_logits.shape[0]):
        semantic = torch.topk(semantic_logits[batch_index], semantic_k).indices.tolist()
        selected = list(semantic)
        selected_set = set(selected)
        density_needed = topk - semantic_k
        for token_index in torch.argsort(
                density_logits[batch_index], descending=True).tolist():
            if token_index not in selected_set and not invalid[batch_index, token_index]:
                selected.append(token_index)
                selected_set.add(token_index)
                density_needed -= 1
                if density_needed == 0:
                    break
        if len(selected) < topk:
            for token_index in torch.argsort(
                    joint_scores[batch_index], descending=True).tolist():
                if token_index not in selected_set and not invalid[batch_index, token_index]:
                    selected.append(token_index)
                    selected_set.add(token_index)
                    if len(selected) == topk:
                        break
        selected_tensor = torch.tensor(selected[:topk], device=class_logits.device)
        order = torch.argsort(joint_scores[batch_index, selected_tensor], descending=True)
        selections.append(selected_tensor[order])
    return torch.stack(selections, dim=0)


def build_query_valid_mask(query_counts):
    if query_counts.ndim != 1 or query_counts.numel() == 0:
        raise ValueError('query_counts must be a non-empty one-dimensional tensor')
    return (torch.arange(int(query_counts.max().item()), device=query_counts.device)[None]
            < query_counts[:, None])
