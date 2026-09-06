#!/usr/bin/env python
import argparse

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Colorize a saved AQBA density-prior array')
    parser.add_argument('--input', required=True, help='.npy density map')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    density = np.load(args.input).squeeze()
    density = (density - density.min()) / max(float(density.max() - density.min()), 1e-8)
    red = (density * 255).astype(np.uint8)
    blue = ((1.0 - density) * 255).astype(np.uint8)
    green = (255 - np.abs(red.astype(np.int16) - blue.astype(np.int16))).astype(np.uint8)
    Image.fromarray(np.stack([red, green, blue], axis=-1)).save(args.output)


if __name__ == '__main__':
    main()
