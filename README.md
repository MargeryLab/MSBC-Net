## MABC-Net: Automic Rectal Tumor Segmentation from MRI Scans

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](readme/fig.jpeg)

## Paper
[MABC-Net: Automic Rectal Tumor Segmentation from MRI Scans](https://arxiv.org/abs/2011.12450)

## Updates
- (02/12/2020) Models and logs(R101_100pro_3x and R101_300pro_3x) are available. 
- (26/11/2020) Models and logs(R50_100pro_3x and R50_300pro_3x) are available.
- (26/11/2020) Higher Performance for Sparse R-CNN is reported by setting the dropout rate as 0.0. 

## Models
Method | inf_time | train_time | box AP | download
--- |:---:|:---:|:---:|:---:
[R50_100pro_3x](projects/MSBC-Net/configs/msbcnet.res50.100pro.3x.yaml) | 23 FPS | 19h  | 42.8 | [model](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) \| [log](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__)
[R50_300pro_3x](projects/MSBC-Net/configs/msbcnet.res50.300pro.3x.yaml) | 22 FPS | 24h  | 45.0 | [model](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) \| [log](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__)
[R101_100pro_3x](projects/MSBC-Net/configs/msbcnet.res101.100pro.3x.yaml) | 19 FPS | 25h  | 44.1 | [model](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) \| [log](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__)
[R101_300pro_3x](projects/MSBC-Net/configs/msbcnet.res101.300pro.3x.yaml) | 18 FPS | 29h  | 46.4 | [model](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) \| [log](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__)

#### Notes
- We observe about 0.3 AP noise.
- The training time is on 8 GPUs with batchsize 16. The inference time is on single GPU. All GPUs are NVIDIA V100.
- We use the models pre-trained on imagenet using torchvision. And we provide [torchvision's ResNet-101.pkl](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) model. 
More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).


## Installation
The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [DETR](https://github.com/facebookresearch/detr).

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install and build libs
```
git clone https://github.com/PeizeSun/SparseR-CNN.git
cd SparseR-CNN
python setup.py build develop
```

2. Link coco dataset path to SparseR-CNN/datasets/coco
```
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

3. Train MSBCNet
```
python projects/MSBCNet/train_net.py --num-gpus 8 \
    --config-file projects/MSBCNet/configs/msbcnet.res50.100pro.3x.yaml
```

4. Evaluate MSBCNet
```
python projects/MSBCNet/train_net.py --num-gpus 8 \
    --config-file projects/MSBCNet/configs/msbcnet.res50.100pro.3x.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

5. Visualize MSBCNet
```    
python demo/demo.py\
    --config-file projects/MSBCNet/configs/msbcnet.res50.100pro.3x.yaml \
    --input path/to/images --output path/to/save_images --confidence-threshold 0.4 \
    --opts MODEL.WEIGHTS path/to/model.pth
```

## License

MSBCNet is released under MIT License.


## Citing

If you use MSBCNet in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@article{peize2020sparse,
  title   =  {{MSBCNet}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}

```
