from datasets import build_dataset
args = type('Args', (), {
        'dataset_file': 'aitodv2',
        'coco_path': 'data/path/AITODv2',
        'device': 'cuda',
        'modelname': 'test01',
        'fix_size': 'store_true',
        'masks': False
    })()


dataset_train = build_dataset(image_set='trainval', args=args)
print(f"Dataset size: {len(dataset_train)}")

# 检查第一个样本
for i in dataset_train:
        img, target = dataset_train[i]
        print(f"Labels: {target['labels']}")
        print(f"Boxes shape: {target['boxes'].shape}")