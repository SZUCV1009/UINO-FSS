# UINO-FSS

**UINO-FSS: Unifying Representation Learning and Few-Shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation**

Wei Zhuo, Zhiyue Tang, Wufeng Xue, Hao Ding, Junkai Ji, and Linlin Shen

*IEEE Transactions on Image Processing*, 2026.

[Paper](https://doi.org/10.1109/TIP.2026.3723708) | [Model Weights](https://drive.google.com/drive/folders/1acBYYjk-UTwcS798ueKlb-rXPQtnCIy_) | [COCO Annotations](https://drive.google.com/drive/folders/17i9FxSG_9yjXaMM8pfiKLH7w2fBzZlgm)

UINO-FSS unifies representation learning and few-shot segmentation through hierarchical distillation and Mamba-based hypercorrelation. It transfers SAM's embedding knowledge into a lightweight bottleneck adapter attached to DINOv2 and generates visual prompts for few-shot segmentation.

This repository provides hierarchical distillation, 1-shot segmentation training, and evaluation code for PASCAL-5i and COCO-20i.

## Installation

### Requirements

- Linux x86_64
- An NVIDIA GPU with a CUDA-compatible driver
- Conda and Git

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.3.1+cu118 |
| torchvision | 0.18.1+cu118 |
| CUDA | 11.8 |
| GCC / G++ | 11.2 |
| NumPy | 2.2.6 |
| Mamba | 2.2.2 |

Run the following steps in Bash, in the order shown. Keep the GPU available while installing the CUDA extensions.

### 1. Clone the repository

```bash
git clone https://github.com/SZUCV1009/UINO-FSS.git
cd UINO-FSS
```

Run all subsequent commands from this repository directory.

### 2. Create the Conda environment

```bash
conda create -n uino-fss --file environment/conda-linux-64.lock -y
conda activate uino-fss
```

The lock file installs Python, the CUDA 11.8 compiler and development libraries, and GCC/G++ 11.2. The NVIDIA driver must already be installed on the host.

### 3. Configure the CUDA toolchain

```bash
source scripts/activate_cuda.sh
```

This selects the CUDA compiler and C++ toolchain in the active Conda environment. Check that CUDA 11.8 and GCC/G++ 11.2 are selected:

```bash
python --version
"$CUDA_HOME/bin/nvcc" --version
"$CXX" --version
nvidia-smi
```

### 4. Install PyTorch and Python dependencies

Install the CUDA 11.8 builds of PyTorch and torchvision first:

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
```

DINOv2 and SAM source modules are included in this repository. xFormers is optional.

### 5. Install the Mamba CUDA backend

```bash
python -m pip install --no-build-isolation --no-deps mamba-ssm==2.2.2 \
  --index-url https://pypi.org/simple
```

This installs `selective_scan_cuda`, which is used by the model's Mamba layers. Keep the environment from Steps 2–3 active during installation.

### 6. Build the VMamba selective-scan extension

```bash
mkdir -p .build
git clone https://github.com/MzeroMiko/VMamba.git .build/VMamba
git -C .build/VMamba checkout --detach 2ed52ead062a51a64521ed3871d52914bf532876

python -m pip install --no-build-isolation --no-deps \
  ./.build/VMamba/kernels/selective_scan
```

This builds `selective_scan_cuda_oflex`. Both this extension and the Mamba backend from Step 5 are required. The build uses the GPU architecture visible during installation. If `.build/VMamba` already exists, reuse it and run the checkout and installation commands.

### 7. Check the installation

```bash
python -m pip check
python scripts/check_environment.py
```

`pip check` checks package dependencies. `check_environment.py` checks GPU execution and both CUDA scan backends. Successful completion prints `Environment checks passed.`.

### Using the environment in a new terminal

```bash
cd /path/to/UINO-FSS
conda activate uino-fss
source scripts/activate_cuda.sh
```

Then follow the model-weight, dataset, training, and evaluation instructions below.

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

The supplied validation splits use `val2014` images. Download `val2014.zip` for evaluation; download `train2014.zip` as well for segmentation training.

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
    ├── train2014/             # Required for training
    │   └── *.jpg
    └── annotations/
        ├── train2014/
        │   └── *.png
        └── val2014/
            └── *.png
```

The PNG masks contain background ID 0 and foreground IDs 1–80. Use them without changing their label values. No JSON-to-PNG conversion is needed when using the supplied masks. Evaluation splits are provided in `splits/coco/val/fold{0,1,2,3}.pkl`.

For both datasets, `--datapath` points to `/path/to/datasets`, the parent of `VOCdevkit/` or `COCO2014/`.

## Training

Run the commands below in Bash from the repository root with the environment above activated.

### Hierarchical distillation

Prepare the SA-1B RGB images using the following layout. The loader uses `sa_000000` through `sa_000009` for training and `sa_000010` for validation. Mask annotations are not needed for embedding distillation.

```text
/path/to/sa_1b/
├── sa_000000/
│   └── *.jpg
├── ...
├── sa_000009/
│   └── *.jpg
└── sa_000010/
    └── *.jpg
```

In addition to the four initialization weights listed above, place the full SAM-H teacher checkpoint `sam_vit_h_4b8939.pth` in `checkpoints/`. Download it from the [SAM project](https://github.com/facebookresearch/segment-anything#model-checkpoints). Obtain SA-1B images from the [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/).

Train the bottleneck adapter progressively at SAM/DINO input sizes of 144/126, 288/252, and 592/518. Each stage loads the preceding stage's output:

```bash
# Stage 1
bash train_distillation.sh 1 0 144 126 4 ./outputs/distill_stage1 '' \
  --dataset_path /path/to/sa_1b \
  --epochs 15 --batch_size 1 --num_workers 4

# Stage 2
bash train_distillation.sh 1 0 288 252 4 ./outputs/distill_stage2 \
  ./outputs/distill_stage1/work_dir/ckpt/iter_final.pth \
  --dataset_path /path/to/sa_1b \
  --epochs 15 --batch_size 1 --num_workers 4

# Stage 3
bash train_distillation.sh 1 0 592 518 4 ./outputs/distill_stage3 \
  ./outputs/distill_stage2/work_dir/ckpt/iter_final.pth \
  --dataset_path /path/to/sa_1b \
  --epochs 15 --batch_size 1 --num_workers 4
```

The first two arguments specify the number of GPUs and their IDs. For two GPUs, use `2 0,1` and set `--batch_size 2` or another multiple of two. Distillation batch size is the total across all GPUs. Adjust it to the available GPU memory. The commands use AdamW with learning rate `1e-3` and weight decay `5e-4`; the DINOv2 encoder is frozen and only the bottleneck adapter is updated.

Each stage saves `epoch_<epoch>.pth` and `iter_final.pth` under `<output>/work_dir/ckpt/`. The final stage output initializes the segmentation model.

### Few-shot segmentation training

Prepare the datasets as described above, including COCO `train2014/` images for COCO training. Training and validation episode lists for all four folds are included under `lists/<benchmark>/fss_list/`.

PASCAL-5i:

```bash
bash train.sh 0 fold0 pascal ./outputs/pascal \
  ./outputs/distill_stage3/work_dir/ckpt/iter_final.pth \
  --datapath /path/to/datasets \
  --bsz 1 --nworker 4 --epochs 100 \
  --load_distilled_model --is_sparse --ce_flag --is_ba --is_pcs \
  --mask_decoder_train_flag
```

COCO-20i:

```bash
bash train.sh 0 fold0 coco ./outputs/coco \
  ./outputs/distill_stage3/work_dir/ckpt/iter_final.pth \
  --datapath /path/to/datasets \
  --bsz 1 --nworker 4 --epochs 50 \
  --load_distilled_model --is_sparse --ce_flag --is_ba --is_pcs \
  --mask_decoder_train_flag
```

The first argument selects the GPU IDs; use `0,1` for two-GPU DDP training. Segmentation `--bsz` is the batch size **per GPU**. Change `fold0` to `fold1`, `fold2`, or `fold3` to train the other folds. Training uses 518 × 518 inputs, AdamW with learning rate `1e-3`, weight decay `0.01`, and polynomial learning-rate decay. The DINOv2 encoder, distilled adapter, and prompt encoder are frozen; the hypercorrelation learner and mask decoder are trained.

Checkpoints and TensorBoard logs are saved under `<output>/<benchmark>/<fold>/`. Evaluate a saved segmentation checkpoint with:

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark pascal --fold 0 --nshot 1 --bsz 1 \
  --checkpoint /path/to/epoch_checkpoint.pth \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

For a short startup run, append `--epochs 1 --max_train_batches 3 --max_val_batches 2` to a training command. Distillation also accepts `--max_train_samples 8` to limit the image subset.

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
