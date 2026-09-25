"""S1-only aspect resize; do not change historical training/evaluation rounding."""
from . import transforms as T
import random


class Native800Resize:
    sizes = [800]
    max_size = 1333

    def __call__(self, image, target):
        width, height = image.size
        scale = min(800 / min(width, height), 1333 / max(width, height))
        # Explicit size avoids the legacy short-edge rounding exceeding max_size.
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return T.resize(image, target, size)


class NativeMultiScaleResize:
    """S2: uniform mild scales without the legacy shrink/crop branch."""
    sizes = [640, 704, 768, 800]
    max_size = 1333

    def __call__(self, image, target):
        width, height = image.size
        short_edge = random.choice(self.sizes)
        scale = min(short_edge / min(width, height), self.max_size / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return T.resize(image, target, size)
