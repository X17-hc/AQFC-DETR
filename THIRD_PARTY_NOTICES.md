# Third-party notices

## 2026-09-07 optional tools and research variants

- Supervision 0.30.1 (MIT): https://github.com/roboflow/supervision/tree/0.30.1 . Optional installed library for Detections, annotation and tiled inference. Original category IDs are mapped explicitly; dataset annotations are not rewritten.
- Dome-DETR (Apache-2.0), inspected commit `2dde3bc1946a3e9fad9abd0612b59fc39bd6b861`: https://github.com/RicePasteM/Dome-DETR . The light density encoder is inspired by depthwise-separable DeFE; spatial coverage is inspired by PAQI's semantic safeguard and spatial allocation motivation. The six-block GN encoder and capped grid quotas are independently implemented variants, not verbatim DeFE/PAQI ports. MWAS, dynamic NMS, HGNetv2 and distribution regression are not imported.
- Hu et al., *Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection*: https://arxiv.org/abs/2505.05741 . Renaming is not an originality claim; results require independent evaluation and attribution.
- faster-coco-eval-aitod 1.0.2 is an optional candidate backend, not installed or validated on Windows Python 3.11. No upstream Linux binary was copied into this environment.

本仓库由 DQ-DETR 衍生研究代码重构，并继承 DINO、Deformable DETR、PyTorch、torchvision、COCO API、AI-TOD API 等第三方组件。各源文件中的原始版权头和许可证声明予以保留；这些组件的许可条款优先于本项目说明。

本次重构中的 AQFC-DETR、AQBA、DGFC 命名不表示对上游实现版权的替代。发布或投稿附带代码前，应逐项核对上游仓库许可证、预训练权重许可与数据集使用条款。
