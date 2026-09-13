"""Shared RGB preprocessing for opt-in standalone inference tools."""
import argparse
import time
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class InferenceRunner:
    def __init__(self, config, checkpoint, device='cuda', amp=True):
        from main import build_model_main
        from util.slconfig import SLConfig
        from util.config_validation import validate_config
        from util.checkpoint import load_native_resume
        values = SLConfig.fromfile(config)._cfg_dict.to_dict()
        validate_config(values)
        values.update(device=device, distributed=False, use_ema=False)
        self.args = argparse.Namespace(**values)
        self.model, _, self.post = build_model_main(self.args)
        load_native_resume(self.model, checkpoint)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.amp = amp and self.device.type == 'cuda'
        self.records = []

    def preprocess(self, image):
        from datasets.transforms import resize
        image = image.convert('RGB') if isinstance(image, Image.Image) else Image.fromarray(image).convert('RGB')
        original_size = (image.height, image.width)
        resized, _ = resize(image, None, max(self.args.data_aug_scales), self.args.data_aug_max_size)
        tensor = TF.normalize(TF.to_tensor(resized), [.485,.456,.406], [.229,.224,.225])
        return tensor.to(self.device), original_size

    @torch.inference_mode()
    def __call__(self, image):
        start = time.perf_counter()
        tensor, size = self.preprocess(image)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        model_start = time.perf_counter()
        with torch.amp.autocast(self.device.type, enabled=self.amp):
            outputs = self.model([tensor])
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter()-model_start)*1000
        result = self.post['bbox'](outputs, torch.tensor([size], device=self.device))[0]
        result = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in result.items()}
        self.records.append(dict(model_ms=model_ms, end_to_end_ms=(time.perf_counter()-start)*1000,
                                 query_count=int(outputs['executed_query_counts'][0]),
                                 input_size=list(tensor.shape[-2:]), original_size=list(size)))
        return result
