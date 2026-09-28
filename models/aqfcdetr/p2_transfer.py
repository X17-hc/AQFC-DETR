"""Layerwise P2 substitutes. Reference execution is explicit, never an inference fallback."""
import torch
from torch import nn
from torch.nn import functional as F
from .precision_modules import ChannelLayerNorm


class P2Substitute(nn.Module):
    def __init__(self, channels=256, width=128):
        super().__init__()
        self.detail = nn.Conv2d(channels, width, 1)
        self.context = nn.Conv2d(channels, width, 1)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.norm = ChannelLayerNorm(width)
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, memory, shapes, padding):
        h2, w2 = map(int, shapes[0]); h3, w3 = map(int, shapes[1])
        n2, n3 = h2*w2, h3*w3
        b, _, c = memory.shape
        p2 = memory[:, :n2].transpose(1, 2).reshape(b,c,h2,w2)
        p3 = memory[:, n2:n2+n3].transpose(1, 2).reshape(b,c,h3,w3)
        m2 = padding[:, :n2].reshape(b,1,h2,w2)
        m3 = padding[:, n2:n2+n3].reshape(b,1,h3,w3)
        detail = self.detail(p2.masked_fill(m2,0)).masked_fill(m2,0)
        context = self.context(p3.masked_fill(m3,0)).masked_fill(m3,0)
        context = F.interpolate(context, (h2,w2), mode='bilinear', align_corners=False)
        z = (detail+context).masked_fill(m2,0)
        residual = self.output(F.silu(self.norm(self.depthwise(z)))).masked_fill(m2,0)
        return (p2+residual).flatten(2).transpose(1,2)


def step_student(encoder, layer_id, memory, pos, refs, shapes, starts, padding):
    """Both branches read precisely the same pre-update memory."""
    cut = int(shapes[0].prod())
    p2 = encoder.p2_substitutes[layer_id-2](memory, shapes, padding)
    low = encoder.layers[layer_id].forward_queries(
        memory[:,cut:], pos[:,cut:], refs[:,cut:], memory, shapes, starts, padding)
    return torch.cat((p2,low),1)


def encoder_forward(encoder, src, pos, shapes, starts, ratios, padding, mode=None, trace=False):
    mode = encoder.p2_transfer_mode if mode is None else mode
    if mode not in ('reference','student'):
        raise ValueError(f'Unknown P2 transfer mode: {mode}')
    refs = encoder.get_reference_points(shapes,ratios,src.device)
    memory = src
    states = []
    for i, layer in enumerate(encoder.layers):
        if mode == 'student' and i >= 2:
            memory = step_student(encoder,i,memory,pos,refs,shapes,starts,padding)
        else:
            memory = layer(memory,pos,refs,shapes,starts,padding)
        if trace:
            states.append(memory)
    if encoder.norm is not None:
        memory = encoder.norm(memory)
    return memory, states


def transfer_loss(student, teacher, weights, valid):
    """Per-pixel teacher RMS, channel-wise SmoothL1/cosine, no padding contribution."""
    with torch.autocast(device_type=student.device.type,enabled=False):
        s,t = student.float(),teacher.detach().float()
        r = t.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-3)
        regression = F.smooth_l1_loss((s-t)/r,torch.zeros_like(s),beta=1.,reduction='none').mean(-1)
        cosine = 1-F.cosine_similarity(s,t,dim=-1,eps=1e-8)
        w = weights.detach().float()*valid.float()
        return ((regression+.25*cosine)*w).sum()/w.sum().clamp_min(1.)


def adaptation_loss(encoder, inputs, targets, phase, reference=None):
    """Original weights must be frozen/eval. Only substitutes participate in optimizer."""
    from .query_allocator import QueryBudgetLoss
    src,pos,shapes,starts,ratios,padding = inputs
    with torch.no_grad():
        if reference is None:
            _, reference = encoder_forward(encoder,*inputs,mode='reference',trace=True)
        cut = int(shapes[0].prod())
        h,w = map(int,shapes[0])
        valid = ~padding[:,:cut]
        heat = QueryBudgetLoss.build_density_targets(targets,(h,w),src.device,
            spatial_valid_mask=valid.reshape(len(src),1,h,w),backend='vectorized')
        weights = 1+4*heat.flatten(1)
    refs = encoder.get_reference_points(shapes,ratios,src.device)
    memory = reference[1].detach()
    losses = []
    for i in range(2,6):
        if phase == 'teacher_forced':
            p2 = encoder.p2_substitutes[i-2](reference[i-1],shapes,padding)
        elif phase == 'free_running':
            memory = step_student(encoder,i,memory,pos,refs,shapes,starts,padding)
            p2 = memory[:,:cut]
        else:
            raise ValueError(f'Unknown adaptation phase: {phase}')
        losses.append(transfer_loss(p2,reference[i][:,:cut],weights,valid))
    return torch.stack(losses).mean(), torch.stack(losses).detach()
