# Modified from https://github.com/Jongchan/attention-module
# v8: tanh-centered ChannelGate + calibrated per-level spatial residual alphas
# Ablation extension: gate_type in ['tanh', 'se'] controls channel attention variant
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, gn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.gn   = nn.GroupNorm(32, out_channel) if gn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn   is not None: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def _logsumexp_2d(tensor):
    flat = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(flat, dim=2, keepdim=True)
    return s + (flat - s).exp().sum(dim=2, keepdim=True).log()


# ============================================================
# ★ [Original / v8] tanh 中心化通道注意力
# ============================================================

class ChannelGateTanh(nn.Module):
    """
    tanh 中心化通道注意力 (v8 核心设计)。

    feat_out = feat × (1 + tanh(sf × mlp_logit))
      logit=0  → feat × 1.00  (中性通道完全不干扰)
      logit>0  → feat × >1.0  (目标通道增强 → FN↓)
      logit<0  → feat × <1.0  (背景通道轻微抑制 → FP↓)

    sf (scale_factor) 可学习，初始 0.1，clamp [0, 0.2]，
    训练过程中逐步增大，最终收敛至最优锐化程度。
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg', 'max'), init_scale_factor=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = list(pool_types)
        self.scale_factor = nn.Parameter(torch.tensor(float(init_scale_factor)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = None
        for pt in self.pool_types:
            if pt == 'avg':
                p = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'max':
                p = F.max_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'lp':
                p = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                stride=(x.size(2), x.size(3)))
            elif pt == 'lse':
                p = _logsumexp_2d(x)
            else:
                raise ValueError(f"Unknown pool_type: {pt}")
            logit = self.mlp(p)
            raw = logit if raw is None else raw + logit

        sf = self.scale_factor.clamp(min=0.0, max=0.2)
        ch_mult = 1.0 + torch.tanh(sf * raw)      # (B, C)，值域(0,2)
        return x * ch_mult.unsqueeze(2).unsqueeze(3)


# ============================================================
# ★ [Ablation] 简单残差 SE 通道注意力  (1 + SE)
# ============================================================

class ChannelGateSE(nn.Module):
    """
    消融对照组：标准 Squeeze-and-Excitation 残差变体。

    公式：feat_out = feat × (1 + sigmoid(mlp(gap(feat))))
    与 tanh 版的核心区别：
      - 激活函数为 sigmoid，输出恒为正 → 只能增强，无法抑制背景通道
      - sigmoid(0) = 0.5 → 中性通道也被放大 ×1.5，引入额外偏置
      - 理论上对背景 FP 的抑制能力弱于 tanh 版

    用于消融实验：验证 tanh 中心化设计对 AP/FP 的具体贡献。
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg',)):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = list(pool_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = None
        for pt in self.pool_types:
            if pt == 'avg':
                p = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'max':
                p = F.max_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            else:
                raise ValueError(f"Unknown pool_type: {pt}")
            logit = self.mlp(p)
            raw = logit if raw is None else raw + logit

        # (1 + SE)：中性logit→×1.5，正logit→×(1+近1)=×2，负logit→×(1+近0)=×1
        ch_mult = 1.0 + torch.sigmoid(raw)   # 值域 (1, 2)，永远>1，无法抑制
        return x * ch_mult.unsqueeze(2).unsqueeze(3)

# ============================================================
# ★ [Ablation] Swish (SiLU) 通道注意力
# ============================================================

class ChannelGateSwish(nn.Module):
    """
    Swish (SiLU) 通道注意力。

    feat_out = feat × (1 + swish(sf × mlp_logit))
      swish(x) = x · σ(x)
      logit=0  → feat × 1.00  (中性通道不干扰)
      logit>0  → feat × >1.0  (目标通道增强)
      logit<0  → feat × >0.72 (背景通道轻微抑制，下界 ≈ 1 - 0.278)

    与 tanh 的区别：
      - Swish 无上界饱和 → 对强激活通道增强更激进
      - Swish 下界约 -0.278 → 抑制能力弱于 tanh（tanh 可到 -1）
      - Swish 是光滑非单调函数，梯度特性优于 tanh
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg', 'max'), init_scale_factor=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = list(pool_types)
        self.scale_factor = nn.Parameter(torch.tensor(float(init_scale_factor)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = None
        for pt in self.pool_types:
            if pt == 'avg':
                p = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'max':
                p = F.max_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'lp':
                p = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                stride=(x.size(2), x.size(3)))
            elif pt == 'lse':
                p = _logsumexp_2d(x)
            else:
                raise ValueError(f"Unknown pool_type: {pt}")
            logit = self.mlp(p)
            raw = logit if raw is None else raw + logit

        sf = self.scale_factor.clamp(min=0.0, max=0.2)
        ch_mult = 1.0 + (sf * raw).sigmoid() * (sf * raw)  # swish = x · σ(x)
        return x * ch_mult.unsqueeze(2).unsqueeze(3)


# ============================================================
# ★ [Ablation] GLU (Gated Linear Unit) 通道注意力
# ============================================================

class ChannelGateGLU(nn.Module):
    """
    GLU 通道注意力。

    MLP 输出 2×gate_channels，等分为 value 和 gate：
      gated = value ⊙ sigmoid(gate)
      ch_mult = 1.0 + sf × gated
      中性 → feat × 1.00；正 → 增强；负 → 抑制。

    与 tanh 的区别：
      - GLU 的"门控"由输入内容决定（data-dependent gating）
      - tanh 对 MLP 输出施加固定形状的非线性；GLU 通过 sigmoid 门自适应调节
      - 理论上有更强的输入条件化能力
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg', 'max'), init_scale_factor=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, 2 * gate_channels),
        )
        self.pool_types = list(pool_types)
        self.scale_factor = nn.Parameter(torch.tensor(float(init_scale_factor)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = None
        for pt in self.pool_types:
            if pt == 'avg':
                p = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'max':
                p = F.max_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'lp':
                p = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                stride=(x.size(2), x.size(3)))
            elif pt == 'lse':
                p = _logsumexp_2d(x)
            else:
                raise ValueError(f"Unknown pool_type: {pt}")
            logit = self.mlp(p)
            value, gate = logit.chunk(2, dim=1)
            gated = value * torch.sigmoid(gate)
            raw = gated if raw is None else raw + gated

        sf = self.scale_factor.clamp(min=0.0, max=0.2)
        ch_mult = 1.0 + sf * raw
        return x * ch_mult.unsqueeze(2).unsqueeze(3)


# ============================================================
# ★ [Ablation] 标准 Sigmoid 通道注意力 (验证 0.5 压制问题)
# ============================================================
class ChannelGateSigmoid(nn.Module):
    """
    消融对照组：标准 Sigmoid 通道注意力 (经典 SE 结构)。

    公式：feat_out = feat × sigmoid(mlp(gap(feat)))
    与 tanh 版和 (1+SE) 版的核心区别：
      - 激活函数直接乘 sigmoid。
      - 当 logit ≈ 0 时，sigmoid(0) = 0.5。这意味着对于不需要特殊处理的中性通道，
        其特征值会被无差别地压制 50%，从而破坏主干网络提取的基础特征分布（即 0.5 压制问题）。
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg',)):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = list(pool_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = None
        for pt in self.pool_types:
            if pt == 'avg':
                p = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'max':
                p = F.max_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            else:
                raise ValueError(f"Unknown pool_type: {pt}")
            logit = self.mlp(p)
            raw = logit if raw is None else raw + logit

        # 标准 Sigmoid：中性logit→×0.5，正logit→接近×1，负logit→接近×0
        ch_mult = torch.sigmoid(raw)
        return x * ch_mult.unsqueeze(2).unsqueeze(3)

# ============================================================
# 空间注意力
# ============================================================

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)], dim=1)


class SpatialAttention(nn.Module):
    """返回空间权重 scale ∈ [0,1]，(B,1,H,W)。"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.compress = ChannelPool()
        self.conv = Conv_BN(2, 1, kernel_size, stride=1,
                            padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(self.compress(x)))


# ============================================================
# DGFC: Density-Guided Feature Calibrator
# ============================================================

class DensityGuidedFeatureCalibrator(nn.Module):
    """
    通道-空间门控特征增强模块。

    gate_type 参数：
      'tanh'    (默认) — 使用 ChannelGateTanh (v8, 论文主方法，中心化1.0)
      'swish'         — 使用 ChannelGateSwish (消融对照：SiLU激活，无上界饱和)
      'glu'           — 使用 ChannelGateGLU (消融对照：门控线性单元，data-dependent gating)
      'se'            — 使用 ChannelGateSE   (消融对照：残差SE，1+SE中心化1.5)
      'sigmoid'       — 使用 ChannelGateSigmoid (消融对照：标准SE，验证0.5压制)
    """

    _DEFAULT_SPATIAL_ALPHAS = [0.00, 0.05, 0.15, 0.10, 0.10]

    def __init__(self,
                 gate_channels: int = 256,
                 reduction_ratio: int = 16,
                 pool_types: list = ('avg', 'max'),
                 use_spatial: bool = True,
                 num_feature_levels: int = 4,
                 level_spatial_alphas: list = None,
                 channel_init_sf: float = 0.1,
                 gate_type: str = 'tanh'):          # ← 新增消融控制参数
        super().__init__()
        self.num_feat = num_feature_levels
        self.use_spatial = use_spatial
        self.gate_type = gate_type

        # ── 通道注意力 (根据 gate_type 选择) ──────────────────────────
        if gate_type == 'tanh':
            self.channel_gate = ChannelGateTanh(
                gate_channels, reduction_ratio, pool_types,
                init_scale_factor=channel_init_sf)
        elif gate_type == 'swish':
            self.channel_gate = ChannelGateSwish(
                gate_channels, reduction_ratio, pool_types,
                init_scale_factor=channel_init_sf)
        elif gate_type == 'glu':
            self.channel_gate = ChannelGateGLU(
                gate_channels, reduction_ratio, pool_types,
                init_scale_factor=channel_init_sf)
        elif gate_type == 'se':
            # 1+SE 消融：只用 avg pool（经典设定）
            self.channel_gate = ChannelGateSE(
                gate_channels, reduction_ratio, pool_types=('avg',))
        elif gate_type == 'sigmoid':
            # 标准 Sigmoid 消融：只用 avg pool（经典设定）
            self.channel_gate = ChannelGateSigmoid(
                gate_channels, reduction_ratio, pool_types=('avg',))
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}. "
                             f"Choose from ['tanh', 'swish', 'glu', 'se', 'sigmoid']")

        # ── 空间注意力 (不变) ──────────────────────────────────────────
        if use_spatial:
            self.spatial_attention = SpatialAttention()

        alphas = list(level_spatial_alphas) if level_spatial_alphas \
                 else list(self._DEFAULT_SPATIAL_ALPHAS)
        while len(alphas) < num_feature_levels:
            alphas.append(alphas[-1])
        self.level_spatial_alphas = alphas[:num_feature_levels]

    def forward(self, x: list, memory: torch.Tensor,
                spatial_shapes: list) -> torch.Tensor:
        feats = []
        idx   = 0
        enc   = memory.transpose(1, 2)
        bs, c, _ = enc.shape

        for i in range(self.num_feat):
            h  = int(spatial_shapes[i][0])
            w  = int(spatial_shapes[i][1])
            hw = h * w

            feat = enc[:, :, idx: idx + hw].view(bs, c, h, w)

            # 残差空间注意力（level-0 跳过）
            alpha = self.level_spatial_alphas[i]
            if self.use_spatial and alpha > 0.0:
                aux = x[i]
                if aux.shape[2:] != (h, w):
                    aux = F.interpolate(aux, size=(h, w),
                                        mode='bilinear', align_corners=False)
                scale = self.spatial_attention(aux)
                feat  = feat * (1.0 + alpha * scale)

            # 通道注意力 (tanh or se，由 gate_type 决定)
            feat = self.channel_gate(feat)

            feats.append(feat.flatten(2).transpose(1, 2))
            idx += hw

        return torch.cat(feats, dim=1)


# ============================================================
# 多尺度特征金字塔（原版不变）
# ============================================================

class DensityPyramidAdapter(nn.Module):
    def __init__(self, channels: int = 256, is_5_scale: bool = False):
        super().__init__()
        self.conv1 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        if is_5_scale:
            self.conv4 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.is_5_scale = is_5_scale

    def forward(self, x: torch.Tensor) -> list:
        out = [x]
        x = self.conv1(x); out.append(x)
        x = self.conv2(x); out.append(x)
        x = self.conv3(x); out.append(x)
        if self.is_5_scale:
            x = self.conv4(x); out.append(x)
        return out
