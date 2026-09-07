import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_to_routing(count, max_objects=1500):
    """Teacher and predicted counts must use identical routing-count units."""
    return (count.float().clamp(min=0) * 1.5 + 50.0).clamp(max=max_objects)


class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel) if gn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def make_density_layers(cfg, in_channels=256, d_rate=2):
    layers = []
    for v in cfg:
        conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        layers.append(conv2d)
        in_channels = v
    return nn.Sequential(*layers)


class AdaptiveQueryBudgetAllocator(nn.Module):
    """Predicts density, ordered count boundaries and a per-image query budget."""

    def __init__(self, feature_dim=256, query_budget_levels=None,
                 max_objects=1500, fallback_queries=900, use_ema=True,
                 ema_decay=0.9997, boundary_warmup_steps=5000):
        super().__init__()
        self.query_budget_levels = list(query_budget_levels or [300, 500, 900, 1500])
        if len(self.query_budget_levels) != 4:
            raise ValueError("AQBA currently requires exactly four query budget levels")
        if self.query_budget_levels != sorted(set(self.query_budget_levels)):
            raise ValueError("query_budget_levels must be strictly increasing")
        self.num_budget_levels = len(self.query_budget_levels)
        self.max_objects = max_objects
        self.fallback_queries = int(fallback_queries)
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Highest-resolution encoder feature -> density-aware representation.
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        self.density_encoder = make_density_layers(
            [512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)

        # Ordered routing-count boundaries.
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

        # Object-count regressor. Routing applies the 1.5*N+50 transform later.
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.budget_pool = nn.AdaptiveAvgPool2d(1)
        self.budget_classifier = nn.Linear(256, self.num_budget_levels)

        # Spatial density prior.
        self.density_head = nn.Conv2d(256, 1, kernel_size=1)

        if self.use_ema:
            self.register_buffer(
                'ema_log_boundaries', torch.log(torch.tensor([60.0, 150.0, 350.0])))
            self.register_buffer('ema_initialized', torch.tensor(False))
            self.register_buffer('boundary_history', torch.zeros(100, 3))
            self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))

        self.register_buffer('training_steps', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = int(boundary_warmup_steps)

        self._init_weights()

    def _init_weights(self):
        for m in self.density_encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        nn.init.normal_(self.boundary_head[-1].weight, std=0.0005)
        nn.init.constant_(self.boundary_head[-1].bias[0], math.log(60.0))
        nn.init.constant_(self.boundary_head[-1].bias[1], -0.164)
        nn.init.constant_(self.boundary_head[-1].bias[2], -0.329)

        nn.init.normal_(self.count_regressor[-1].weight, std=0.0005)
        nn.init.constant_(self.count_regressor[-1].bias, 3.5)

        nn.init.normal_(self.density_head.weight, std=0.01)
        nn.init.constant_(self.density_head.bias, -2.19)

    def forward(self, feature_map, spatial_shapes=None, real_counts=None, epoch=0):
        if feature_map.dim() == 3:
            if spatial_shapes is None:
                raise ValueError("spatial_shapes needed when feature_map is 3D")
            bs, l, c = feature_map.shape
            h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)

        bs, c, h, w = feature_map.shape
        device = feature_map.device

        x = self.density_conv1(feature_map)
        density_feat = self.density_encoder(x)

        bd_feat = self.boundary_pool(density_feat).flatten(1)
        # Nonlinear count/boundary operations need FP32 even under autocast.
        raw_out = self.boundary_head(bd_feat).float()

        warmup_factor = self._get_warmup_factor()

        log_b1 = raw_out[:, 0].clamp(min=math.log(30.0), max=math.log(150.0))

        min_log_gap = 0.3
        delta12 = F.softplus(raw_out[:, 1].clamp(max=20)) + min_log_gap
        delta23 = F.softplus(raw_out[:, 2].clamp(max=20)) + min_log_gap

        log_b2 = log_b1 + delta12
        log_b2 = log_b2.clamp(max=math.log(500.0))

        log_b3 = log_b2 + delta23

        log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
        raw_log_boundaries = log_boundaries
        valid_boundaries = (torch.isfinite(raw_out).all(1) &
                            torch.isfinite(log_boundaries).all(1) &
                            (log_boundaries[:, 1:] > log_boundaries[:, :-1]).all(1))
        default_boundaries = torch.log(torch.tensor([60., 150., 350.], device=device))
        # Validate BEFORE EMA updates: a bad sample must never poison the buffer.
        if self.use_ema and self.training and valid_boundaries.any():
            with torch.no_grad():
                mean = log_boundaries[valid_boundaries].mean(0)
                if not self.ema_initialized or not torch.isfinite(self.ema_log_boundaries).all():
                    self.ema_log_boundaries.copy_(mean)
                    self.ema_initialized.fill_(True)
                else:
                    self.ema_log_boundaries.lerp_(mean, 1.0 - self.ema_decay)
                self._update_boundary_history(mean)
        if self.use_ema and bool(self.ema_initialized):
            log_boundaries_for_use = self.ema_log_boundaries.unsqueeze(0).expand(bs, -1)
        else:
            log_boundaries_for_use = log_boundaries

        ordered = (log_boundaries_for_use[:, 1:] > log_boundaries_for_use[:, :-1]).all(dim=1)
        finite_boundaries = torch.isfinite(log_boundaries_for_use).all(dim=1)
        invalid_boundaries = ~(ordered & finite_boundaries & valid_boundaries)
        if invalid_boundaries.any():
            fallback = default_boundaries
            if self.use_ema and torch.isfinite(self.ema_log_boundaries).all() and (
                    self.ema_log_boundaries[1:] > self.ema_log_boundaries[:-1]).all():
                fallback = self.ema_log_boundaries
            log_boundaries_for_use = torch.where(
                invalid_boundaries[:, None], fallback[None], log_boundaries_for_use)

        boundaries = torch.exp(log_boundaries_for_use)

        raw_count = self.count_regressor(density_feat).squeeze(1).float()
        # Clamp BEFORE exp; retain the raw finite flag so Inf cannot masquerade as 1500.
        finite_raw_count = torch.isfinite(raw_count)
        pred_count = raw_count.clamp(0., math.log(self.max_objects)).exp()
        pred_route = count_to_routing(pred_count, self.max_objects)

        budget_feature = self.budget_pool(density_feat).flatten(1)
        budget_logits = self.budget_classifier(budget_feature)
        budget_probabilities = torch.softmax(budget_logits, dim=-1)

        teacher_ratio = self.teacher_ratio(epoch) if self.training else 0.0
        if real_counts is not None:
            real_counts = torch.as_tensor(real_counts, device=device, dtype=torch.float32)
            gt_route = count_to_routing(real_counts, self.max_objects)
            routing_count = teacher_ratio * gt_route + (1.0 - teacher_ratio) * pred_route
        else:
            routing_count = pred_route

        finite_count = finite_raw_count & torch.isfinite(routing_count)
        routing_count = torch.where(
            finite_count, routing_count,
            torch.full_like(routing_count, float(self.fallback_queries)))
        budget_indices = self._assign_query_levels(routing_count, boundaries)
        levels = torch.tensor(self.query_budget_levels, device=device, dtype=torch.long)
        query_counts = levels[budget_indices]
        if not finite_count.all():
            fallback_level_index = min(
                range(len(self.query_budget_levels)),
                key=lambda index: abs(self.query_budget_levels[index] - self.fallback_queries))
            budget_indices = torch.where(
                finite_count, budget_indices,
                torch.full_like(budget_indices, fallback_level_index))
            query_counts = torch.where(
                finite_count, query_counts,
                torch.full_like(query_counts, self.query_budget_levels[fallback_level_index]))

        density_logits = self.density_head(density_feat).float()
        density_prior = torch.sigmoid(density_logits.clamp(-10, 10))
        density_peaks = self._generate_density_peaks(density_prior, h, w)

        if self.training:
            self.training_steps.add_(1)

        return {
            'boundaries': boundaries,
            'log_boundaries': log_boundaries_for_use,
            'raw_log_boundaries': raw_log_boundaries,
            'log_boundaries_ema': log_boundaries_for_use,
            'predicted_count': pred_count,
            'raw_count': raw_count,
            'budget_logits': budget_logits,
            'budget_probabilities': budget_probabilities,
            'density_feature': density_feat,
            'density_prior': density_prior,
            'density_logits': density_logits,
            'density_peaks': density_peaks,
            'query_counts': query_counts,
            'budget_indices': budget_indices,
            'routing_count': routing_count,
            'teacher_ratio': teacher_ratio,
            'invalid_fallback_count': (~finite_count).sum(),
            'invalid_boundary_fallback_count': invalid_boundaries.sum(),
            'warmup_factor': warmup_factor,
        }

    @staticmethod
    def teacher_ratio(epoch):
        if epoch <= 2:
            return 1.0
        return {3: 0.75, 4: 0.50, 5: 0.25}.get(int(epoch), 0.0)

    def _get_warmup_factor(self) -> float:
        if not self.training:
            return 1.0
        steps = self.training_steps.item()
        if steps >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + torch.cos(torch.tensor((1 - steps / self.warmup_steps) * 3.14159))).item()

    def _smooth_boundaries_for_longtail(self, log_boundaries: torch.Tensor, real_counts: torch.Tensor) -> torch.Tensor:
        if real_counts is None:
            return log_boundaries
        log_boundaries = log_boundaries.clone()
        extreme_mask = real_counts > 100
        if extreme_mask.any():
            if self.history_ptr > 10:
                hist_mean = self.boundary_history[:self.history_ptr].mean(dim=0)
                log_boundaries[extreme_mask] = 0.7 * log_boundaries[extreme_mask] + 0.3 * hist_mean
        return log_boundaries

    def _compute_dynamic_ema_decay(self, real_counts: torch.Tensor) -> float:
        if real_counts is None:
            return self.ema_decay
        max_count = real_counts.max().item()
        if max_count > 150:
            return 0.999
        elif max_count > 80:
            return 0.998
        else:
            return self.ema_decay

    def _update_boundary_history(self, boundaries: torch.Tensor):
        ptr = self.history_ptr.item()
        self.boundary_history[ptr % 100] = boundaries
        self.history_ptr += 1

    def _compute_soft_weights(self, N_eval, log_boundaries):
        temperature = 1.0
        log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)
        c0 = log_boundaries[:, 0] - 0.5
        c1 = (log_boundaries[:, 0] + log_boundaries[:, 1]) / 2
        c2 = (log_boundaries[:, 1] + log_boundaries[:, 2]) / 2
        c3 = log_boundaries[:, 2] + 0.5
        centers = torch.stack([c0, c1, c2, c3], dim=1)
        distances = -torch.abs(log_N - centers)
        soft_weights = F.softmax(distances / temperature, dim=1)
        return soft_weights

    def _assign_query_levels(self, N_eval, boundaries):
        bs = N_eval.shape[0]
        level_indices = torch.zeros(bs, dtype=torch.long, device=N_eval.device)
        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
        level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
        level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
        level_indices[N_eval >= b3] = 3
        return level_indices

    def _generate_density_peaks(self, density_prior, h, w):
        bs = density_prior.shape[0]
        max_k = max(self.query_budget_levels)
        heatmap_flat = density_prior.flatten(2).squeeze(1)
        actual_k = min(h * w, max_k)
        _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
        topk_y = (topk_ind // w).float() + 0.5
        topk_x = (topk_ind % w).float() + 0.5
        peaks = torch.stack([
            (topk_x / w).clamp(0.01, 0.99),
            (topk_y / h).clamp(0.01, 0.99)
        ], dim=-1)
        if actual_k < max_k:
            padding = density_prior.new_zeros(bs, max_k - actual_k, 2)
            peaks = torch.cat([peaks, padding], dim=1)
        return peaks


class QueryBudgetLoss(nn.Module):
    def __init__(self,
                 coverage_weight=0.5,
                 spacing_weight=1.0,
                 count_weight=0.2,
                 interval_weight=0.25,
                 boundary_guide_weight=1.2,
                 density_weight=0.25,
                 enable_adaptive_targets=True,
                 enable_loss_clipping=True):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.spacing_weight = spacing_weight
        self.count_weight = count_weight
        self.interval_weight = interval_weight
        self.boundary_guide_weight = boundary_guide_weight
        self.density_weight = density_weight
        self.enable_adaptive_targets = enable_adaptive_targets
        self.enable_loss_clipping = enable_loss_clipping
        self.smooth_l1 = nn.SmoothL1Loss()

        self.register_buffer('default_target_coverage',
                             torch.tensor([0.40, 0.70, 0.90]))
        self.register_buffer('default_target_boundaries_log',
                             torch.log(torch.tensor([60.0, 150.0, 350.0])))

    def _compute_adaptive_targets(self, real_counts, device):
        # Boundaries and coverage are population-level routing statistics, not
        # per-object pixel sizes. Keep the target in routing-count units.
        batch_size = real_counts.shape[0]
        target_boundaries_log = self.default_target_boundaries_log.to(device).expand(batch_size, -1)
        target_coverage = self.default_target_coverage.to(device).expand(batch_size, -1)
        return target_boundaries_log, target_coverage

    def forward(self, outputs, targets):
        device = outputs['boundaries'].device
        if isinstance(targets, dict) and 'real_counts' in targets:
            real_counts = targets['real_counts'].to(device)
        else:
            real_counts = targets.to(device)
        real_counts = real_counts.float().clamp(min=0.0)
        bs = real_counts.shape[0]
        log_b = outputs.get('raw_log_boundaries', outputs['log_boundaries']).float()
        log_object_count = torch.log(real_counts.clamp(min=1.0))
        route_count = count_to_routing(real_counts)
        log_route_count = torch.log(route_count)
        if self.enable_adaptive_targets:
            target_boundaries_log, target_coverage = self._compute_adaptive_targets(real_counts, device)
        else:
            target_boundaries_log = self.default_target_boundaries_log.unsqueeze(0).expand(bs, -1)
            target_coverage = self.default_target_coverage.unsqueeze(0).expand(bs, -1)
        tau = 1.0
        cdf = torch.stack([
            torch.sigmoid((log_b[:, index] - log_route_count) / tau).mean()
            for index in range(3)
        ])
        loss_coverage = F.mse_loss(cdf, self.default_target_coverage.to(device))
        loss_boundary_guide = F.smooth_l1_loss(log_b, target_boundaries_log)

        loss_spacing = (
                F.relu(math.log(30.0) - log_b[:, 0]) * 3.0 +
                F.relu(log_b[:, 0] + 0.3 - log_b[:, 1]) * 3.0 +
                F.relu(log_b[:, 1] + 0.3 - log_b[:, 2]) * 3.0
        ).mean()

        target_boundaries = torch.exp(target_boundaries_log)
        target_intervals = (
            (route_count[:, None] >= target_boundaries).long().sum(dim=1)
        ).clamp(max=outputs['budget_logits'].shape[1] - 1)
        loss_interval = F.cross_entropy(outputs['budget_logits'].float(), target_intervals)
        loss_count = self.smooth_l1(outputs['raw_count'].float().reshape(-1), log_object_count)
        loss_density_map = outputs['density_prior'].sum() * 0.0
        detection_targets = targets.get('targets') if isinstance(targets, dict) else None
        if detection_targets is not None:
            density_target, density_valid_mask = self.build_density_targets(
                detection_targets, outputs['density_prior'].shape[-2:], device,
                return_valid_mask=True, spatial_valid_mask=outputs.get('density_valid_mask'))
            loss_density_map = self.density_focal_loss(
                outputs['density_prior'], density_target, density_valid_mask)
        warmup_factor = outputs.get('warmup_factor', 1.0)
        total_loss = (
                self.coverage_weight * loss_coverage * warmup_factor +
                self.spacing_weight * loss_spacing +
                self.count_weight * loss_count +
                self.interval_weight * loss_interval +
                self.boundary_guide_weight * loss_boundary_guide +
                self.density_weight * loss_density_map
        )
        result = {
            'loss_allocator_total': total_loss,
            'coverage_rates': cdf,
            'boundary_vals': torch.exp(log_b).mean(dim=0),
            'loss_budget_coverage': loss_coverage,
            'loss_budget_interval': loss_interval,
            'loss_count': loss_count,
            'loss_budget_spacing': loss_spacing,
            'loss_boundary_guide': loss_boundary_guide,
            'loss_density_map': loss_density_map,
        }
        if 'log_boundaries_ema' in outputs:
            result['boundary_vals_ema'] = torch.exp(outputs['log_boundaries_ema']).mean(dim=0)
        if self.enable_adaptive_targets:
            result['adaptive_target_boundaries'] = torch.exp(target_boundaries_log).mean(dim=0)
            result['adaptive_target_coverage'] = target_coverage.mean(dim=0)
        return result

    @staticmethod
    def build_density_targets(targets, spatial_size, device, return_valid_mask=False,
                              spatial_valid_mask=None):
        """Build CenterNet-style Gaussian center maps from normalized cxcywh boxes."""
        height, width = int(spatial_size[0]), int(spatial_size[1])
        heatmaps = torch.zeros(len(targets), 1, height, width, device=device)
        valid_mask = torch.zeros_like(heatmaps, dtype=torch.bool)
        if spatial_valid_mask is not None:
            if tuple(spatial_valid_mask.shape) != tuple(heatmaps.shape):
                raise ValueError('density_valid_mask must match the density map shape')
            valid_mask = spatial_valid_mask.to(device=device, dtype=torch.bool).clone()
        sizes = [target.get('size') for target in targets]
        sized = [size for size in sizes if size is not None]
        max_height = max((float(size[0]) for size in sized), default=1.0)
        max_width = max((float(size[1]) for size in sized), default=1.0)
        for batch_index, target in enumerate(targets):
            size = target.get('size')
            if spatial_valid_mask is not None:
                valid_height = int(valid_mask[batch_index, 0].any(dim=1).sum())
                valid_width = int(valid_mask[batch_index, 0].any(dim=0).sum())
                if valid_height == 0 or valid_width == 0:
                    raise ValueError('Density supervision requires non-empty encoder cells')
            elif size is None:
                valid_height, valid_width = height, width
            else:
                valid_height = max(1, min(height, round(float(size[0]) / max_height * height)))
                valid_width = max(1, min(width, round(float(size[1]) / max_width * width)))
            if spatial_valid_mask is None:
                valid_mask[batch_index, :, :valid_height, :valid_width] = True
            boxes = target.get('boxes')
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device)
            for cx, cy, box_w, box_h in boxes:
                center_x = int(torch.clamp(cx * valid_width, 0, valid_width - 1).item())
                center_y = int(torch.clamp(cy * valid_height, 0, valid_height - 1).item())
                radius = int(torch.clamp(
                    torch.round(0.5 * torch.maximum(
                        box_w * valid_width, box_h * valid_height)),
                    1, 4).item())
                x0, x1 = max(0, center_x - radius), min(valid_width, center_x + radius + 1)
                y0, y1 = max(0, center_y - radius), min(valid_height, center_y + radius + 1)
                yy = torch.arange(y0, y1, device=device, dtype=torch.float32) - center_y
                xx = torch.arange(x0, x1, device=device, dtype=torch.float32) - center_x
                sigma = max(radius / 3.0, 1.0 / 3.0)
                gaussian = torch.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) /
                                     (2.0 * sigma * sigma))
                current = heatmaps[batch_index, 0, y0:y1, x0:x1]
                heatmaps[batch_index, 0, y0:y1, x0:x1] = torch.maximum(current, gaussian)
        return (heatmaps, valid_mask) if return_valid_mask else heatmaps

    @staticmethod
    def density_focal_loss(prediction, target, valid_mask=None):
        # Half/bfloat16 round 1-1e-6 to 1, causing log(0) and 0*Inf=NaN.
        prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
        target = target.float()
        positive = target.eq(1.0).float()
        negative = target.lt(1.0).float()
        negative_weight = (1.0 - target).pow(4)
        positive_loss = -(prediction.log()) * (1.0 - prediction).pow(2) * positive
        negative_loss = -(1.0 - prediction).log() * prediction.pow(2) * negative_weight * negative
        if valid_mask is not None:
            valid = valid_mask.to(prediction.dtype)
            positive_loss = positive_loss * valid
            negative_loss = negative_loss * valid
            positive = positive * valid
        positive_count = positive.sum().clamp(min=1.0)
        return (positive_loss.sum() + negative_loss.sum()) / positive_count
