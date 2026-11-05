# UINO-FSS

**UINO-FSS: Unifying Representation Learning and Few-Shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation**

Wei Zhuo, Zhiyue Tang, Wufeng Xue, Hao Ding, Junkai Ji, and Linlin Shen

*IEEE Transactions on Image Processing*, 2026.

[Paper](https://doi.org/10.1109/TIP.2026.3723708) | [Model Weights](https://drive.google.com/drive/folders/1acBYYjk-UTwcS798ueKlb-rXPQtnCIy_) | [COCO Annotations](https://drive.google.com/drive/folders/17i9FxSG_9yjXaMM8pfiKLH7w2fBzZlgm)

UINO-FSS unifies representation learning and few-shot segmentation through hierarchical distillation and Mamba-based hypercorrelation. It transfers SAM's embedding knowledge into a lightweight bottleneck adapter attached to DINOv2 and generates visual prompts for few-shot segmentation.

This repository provides 1-shot evaluation code for PASCAL-5i and COCO-20i.

## Installation

### Requirements

- Linux x86_64
- An NVIDIA GPU with a CUDA-compatible driver
- Conda and Git

The environment uses Python 3.10, PyTorch 2.3.1, torchvision 0.18.1, CUDA 11.8, and GCC/G++ 11.2. The Conda lock file includes the CUDA compiler, development libraries, and C++ toolchain.

### Environment setup

```bash
git clone https://github.com/SZUCV1009/UINO-FSS.git
cd UINO-FSS

conda create -n uino-fss --file environment/conda-linux-64.lock -y
conda activate uino-fss
source scripts/activate_cuda.sh

python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --index-url https://pypi.org/simple

python -m pip install --no-build-isolation --no-deps mamba-ssm==2.2.2 \
  --index-url https://pypi.org/simple

mkdir -p .build
git clone https://github.com/MzeroMiko/VMamba.git .build/VMamba
git -C .build/VMamba checkout --detach 2ed52ead062a51a64521ed3871d52914bf532876
python -m pip install --no-build-isolation --no-deps \
  ./.build/VMamba/kernels/selective_scan

python -m pip check
```

Keep the GPU visible while building the CUDA extensions. Both `mamba-ssm` and VMamba's selective-scan extension are required. DINOv2 and SAM source modules are included in this repository; xFormers is optional.

In a new terminal, activate the environment and configure its CUDA toolchain before running:

```bash
conda activate uino-fss
source scripts/activate_cuda.sh
```

Optionally check the installation with:

```bash
python scripts/check_environment.py
```

## Model Weights

Download the pretrained and final model weights from [Google Drive](https://drive.google.com/drive/folders/1acBYYjk-UTwcS798ueKlb-rXPQtnCIy_). Request access from the folder owner if prompted.

Place the files as follows:

```text
UINO-FSS/
├── checkpoints/
│   ├── dinov2_vitb14_pretrain.pth
│   ├── sam_vit_b_01ec64.pth
│   ├── sam_maskDecoder_H.pth
│   └── sam_promptEncoder_H.pth
├── final_pascal/
│   ├── fold0.pth
│   ├── fold1.pth
│   ├── fold2.pth
│   └── fold3.pth
└── final_coco/
    ├── fold0.pth
    ├── fold1.pth
    ├── fold2.pth
    └── fold3.pth
```

Each evaluation requires the four initialization files in `checkpoints/` and the final checkpoint for the selected benchmark and fold. Separate distilled weights and training intermediate checkpoints are not required.

## Dataset Preparation

### PASCAL-5i

Prepare PASCAL VOC 2012 images and augmented segmentation labels in the following structure:

```text
/path/to/datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        │   └── *.jpg
        └── SegmentationClassAug/
            └── *.png
```

Use the provided `splits/pascal/val/fold{0,1,2,3}.txt` files. The loader expects augmented labels under `SegmentationClassAug/`.

### COCO-20i

Download the original COCO 2014 images from the [official COCO website](https://cocodataset.org/#download), and download the preprocessed semantic masks below:

| Download | Contents |
| --- | --- |
| [COCO semantic masks](https://drive.google.com/file/d/1pczbyAjScA5yqlRzF8rkbuAKvYJKyQUG/view) | 82,081 training masks and 40,137 validation masks; 367 MiB |
| [SHA-256 checksum](data/COCO2014-semantic-masks.sha256) | Archive checksum |

The supplied validation splits use `val2014` images. Download `val2014.zip` for evaluation; `train2014.zip` is optional for the current evaluation commands.

From the repository root, extract the image archive:

```bash
DATA_ROOT=/path/to/datasets
mkdir -p "$DATA_ROOT/COCO2014"
python -m zipfile -e /path/to/downloads/val2014.zip "$DATA_ROOT/COCO2014"
```

Place `COCO2014-semantic-masks.tar.gz` in the repository root, then verify and extract it:

```bash
sha256sum -c data/COCO2014-semantic-masks.sha256
tar -xzf COCO2014-semantic-masks.tar.gz -C "$DATA_ROOT"
```

The resulting structure should be:

```text
/path/to/datasets/
└── COCO2014/
    ├── val2014/
    │   └── *.jpg
    ├── train2014/             # Optional RGB images
    │   └── *.jpg
    └── annotations/
        ├── train2014/
        │   └── *.png
        └── val2014/
            └── *.png
```

The PNG masks contain background ID 0 and foreground IDs 1–80. Use them without changing their label values. No JSON-to-PNG conversion is needed when using the supplied masks. Evaluation splits are provided in `splits/coco/val/fold{0,1,2,3}.pkl`.

For both datasets, `--datapath` points to `/path/to/datasets`, the parent of `VOCdevkit/` or `COCO2014/`.

## Evaluation

Run commands from the repository root. The following settings use DINOv2-B, an input resolution of 518 × 518, and one support image per episode.

### PASCAL-5i

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark pascal --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

### COCO-20i

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark coco --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

The checkpoint is selected automatically as `final_<benchmark>/fold<fold>.pth`. To specify another checkpoint, add `--checkpoint /path/to/model.pth`.

### Four-fold evaluation

```bash
for fold in 0 1 2 3; do
  python test.py \
    --datapath /path/to/datasets \
    --benchmark pascal --fold "$fold" --nshot 1 --bsz 1 \
    --img_size 518 --stop_layer 4 \
    --is_sparse --ce_flag --is_ba --is_pcs
done
```

Use `--benchmark coco` for COCO-20i. Report the average of the four final fold mIoU values.

Set `CUDA_VISIBLE_DEVICES` to select a GPU. Logs are saved under `log_test/<benchmark>/<fold>/`; use `--logpath` to change the output directory. Add `--visualize` to save predictions under `vis/`.

## Results

Performance on PASCAL-5i and COCO-20i (mIoU, %):

| Dataset | Setting | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PASCAL-5i | 1-shot | 80.7 | 83.2 | 77.8 | 80.5 | 80.6 |
| PASCAL-5i | 5-shot | 84.4 | 87.3 | 79.0 | 85.5 | 84.1 |
| COCO-20i | 1-shot | 62.2 | 66.0 | 65.5 | 64.2 | 64.5 |
| COCO-20i | 5-shot | 66.6 | 73.3 | 70.9 | 68.1 | 69.7 |

## Citation

```bibtex
@article{zhuo2026uinofss,
  title   = {{UINO-FSS}: Unifying Representation Learning and Few-Shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation},
  author  = {Zhuo, Wei and Tang, Zhiyue and Xue, Wufeng and Ding, Hao and Ji, Junkai and Shen, Linlin},
  journal = {IEEE Transactions on Image Processing},
  volume  = {35},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3723708}
}
```

## Acknowledgements

This project builds on [DINOv2](https://github.com/facebookresearch/dinov2), [Segment Anything](https://github.com/facebookresearch/segment-anything), [Mamba](https://github.com/state-spaces/mamba), and [VMamba](https://github.com/MzeroMiko/VMamba). We thank the authors for releasing their code and models. Third-party notices are available in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
