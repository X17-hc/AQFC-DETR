#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description='Draw COCO-format predictions')
    parser.add_argument('--image', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--image-id', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--score-threshold', type=float, default=0.3)
    args = parser.parse_args()
    image = Image.open(args.image).convert('RGB')
    draw = ImageDraw.Draw(image)
    predictions = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    for prediction in predictions:
        if prediction.get('image_id') != args.image_id or prediction.get('score', 0) < args.score_threshold:
            continue
        x, y, width, height = prediction['bbox']
        draw.rectangle((x, y, x + width, y + height), outline='red', width=2)
        draw.text((x, y), f"{prediction.get('category_id')} {prediction.get('score'):.2f}", fill='yellow')
    image.save(args.output)


if __name__ == '__main__':
    main()
