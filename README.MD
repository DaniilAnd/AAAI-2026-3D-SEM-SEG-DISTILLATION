# AAAI 2026: 3D Semantic Segmentation with Knowledge Distillation for Dynamic-Size Feature Maps



![Poster](./images/distill_process_drawio.png)

### Abstract

This paper presents a plug-in framework for improving LiDAR-based 3D semantic segmentation. By aggregating multiple frames, we reconstruct denser and more complete point clouds, which act as supervision for a knowledge distil- lation process invariant to feature-map dimensionality. The student model thus benefits from enriched representations while maintaining standard single-frame inference. On the large-scale Waymo Open Dataset, our approach improves segmentation performance by 0.5–3% in mIoU and 0.4-1.1% in overall accuracy (allAcc) compared to the baseline student model, demonstrating consistent gains on challenging out- door scenes.

## Demo

All videos are hosted on google platform.

- [Demo AAAI 2026](https://drive.google.com/drive/folders/1XVmy2st5FB3VIO_Rh1TlWaMjpohc2D2y?usp=sharing)


# Waymo

## Prepare dataset

We using  [Pointcept](https://github.com/Pointcept/Pointcept) as main framework for training model 

1. Download Pointcept and setup requirements
2. Prepare dataset Waymo (we use v1.4.2)

```
python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py --dataset_root your_path --output_root your_path--splits training validation --num_workers N
```

3. Patch dataset Waymo

```
python patch_scene_pointcept.py --dataset pointcept --dataroot your_path --num_workers N
```

![Waymo structure](./images/structure_waymo.png)


## Acknowledgements

- [`Pointcept`](https://github.com/open-mmlab/OpenPCDet) is used for training supervised models.