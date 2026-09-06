# swa_model.py
import torch
from collections import OrderedDict

checkpoints = [
    '/workspace/DQDetr/logs/DQDETR_ver6_CopyPaste/0330-NewRecord/checkpoint0020.pth',
    '/workspace/DQDetr/logs/DQDETR_ver6_CopyPaste/0330-NewRecord/checkpoint0021.pth',
    '/workspace/DQDetr/logs/DQDETR_ver6_CopyPaste/0330-NewRecord/checkpoint0022.pth',
    '/workspace/DQDetr/logs/DQDETR_ver6_CopyPaste/0330-NewRecord/checkpoint0023.pth',  # 即最终0.315的pth
]


def swa_checkpoints(ckpt_paths):
    avg_state = None
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location='cpu')
        # 优先用ema_model
        state = ckpt.get('ema_model', ckpt['model'])
        if avg_state is None:
            avg_state = {k: v.float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    n = len(ckpt_paths)
    for k in avg_state:
        avg_state[k] /= n
    return avg_state


swa_state = swa_checkpoints(checkpoints)
torch.save({'model': swa_state, 'ema_model': swa_state}, 'swa_checkpoint.pth')