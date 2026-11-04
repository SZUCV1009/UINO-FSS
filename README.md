# UINO-FSS

**UINO-FSS: Unifying Representation Learning and Few-Shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation**

Wei Zhuo, Zhiyue Tang, Wufeng Xue, Hao Ding, Junkai Ji, and Linlin Shen
*IEEE Transactions on Image Processing*, vol. 35, 2026.

[Paper](https://doi.org/10.1109/TIP.2026.3723708) · [Model weights — Google Drive](https://drive.google.com/drive/folders/1acBYYjk-UTwcS798ueKlb-rXPQtnCIy_)

UINO-FSS combines a DINOv2 encoder with a lightweight segmenter. Hierarchical distillation aligns early DINOv2 representations with SAM embeddings through a bottleneck adapter. Semantic-aware visual prompts and Mamba-based hypercorrelation guide few-shot segmentation.

This release documents **1-shot evaluation on PASCAL-5i and COCO-20i**. Training instructions are not included. The paper also reports 5-shot and FSS-1000 results; those evaluation paths are not supported by this release.

## 1. Environment

### Reference environment

The reference is the existing Linux `dinov2` Conda environment used for the recorded Pascal evaluation:

| Component | Installed version |
| --- | --- |
| Python | 3.10.18 |
| PyTorch | 2.3.1+cu118 |
| torchvision | 0.18.1+cu118 |
| PyTorch CUDA runtime | 11.8 |
| Conda `cuda-nvcc` | 11.8.89 |
| NumPy | 2.2.6 |
| Albumentations | 2.0.8 |
| OpenCV headless | 4.11.0.86 |
| timm | 1.0.15 |
| OmegaConf | 2.3.0 |
| einops | 0.8.1 |
| Triton | 2.3.1 |
| VMamba `selective_scan` | 0.0.2 |

An NVIDIA GPU and a compatible driver are required: evaluation utilities call `.cuda()` directly. CPU-only evaluation is not supported. No minimum GPU memory claim is made for this release.

[`requirements.txt`](requirements.txt) pins the installed versions of the evaluation dependencies and their active package dependencies. [`environment/dinov2-packages.txt`](environment/dinov2-packages.txt) records the full local Python package snapshot, including packages unrelated to evaluation. The snapshot is a reference, not a standalone installation file.

### Installation

```bash
git clone https://github.com/SZUCV1009/UINO-FSS.git
cd UINO-FSS

conda create -n uino-fss python=3.10.18 pip=25.1 setuptools=78.1.1 wheel=0.45.1 -y
conda activate uino-fss

python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
```

The repository contains the DINOv2 and SAM source modules needed by this entry point; separate installations of those repositories are unnecessary. Run evaluation commands from the repository root.

These pins now match the existing environment, including **NumPy 2.2.6**. Installation into a fresh environment and full GPU evaluation of this release have **not** been repeated. Package versions alone do not reproduce the GPU driver, compiler, or compiled extension binaries. The earlier README's NumPy 1.26.4 substitution has been removed.

### Selective-scan extension

For the accelerated backend, install a compatible CUDA toolkit and C++ compiler. The reference Conda environment contains CUDA `nvcc` 11.8.89:

```bash
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8.89 -y
nvcc --version
```

VMamba's CUDA selective-scan extension is installed separately from `requirements.txt`. Build it with PyTorch already installed and an NVIDIA GPU available:

```bash
git clone https://github.com/MzeroMiko/VMamba.git /tmp/uino-fss-vmamba
python -m pip install --no-build-isolation /tmp/uino-fss-vmamba/kernels/selective_scan
python -c "import selective_scan_cuda_oflex; print('Selective scan CUDA extension: OK')"
```

Source: [VMamba selective-scan kernels](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan). This command follows current upstream source; the exact source commit used to build the local extension was not recorded. A successful build and import on the target machine must be checked. Rebuild after changing PyTorch/CUDA. Installing `cuda-nvcc` alone does not guarantee that every build-tool requirement is satisfied.

The bundled implementation has a slower PyTorch fallback when no supported CUDA scan extension is importable. `mamba-ssm==2.2.2` and `causal-conv1d==1.4.0` exist in the full reference snapshot, but are not directly imported by this evaluation path; they are not required when using the VMamba extension above. xFormers is optional for this configuration; the recorded Pascal run completed without it.

Check the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
python test.py --help
```

## 2. Download model weights

Download from **[Google Drive — UINO-FSS model weights](https://drive.google.com/drive/folders/1acBYYjk-UTwcS798ueKlb-rXPQtnCIy_)**.

On November 4, 2026, all **34 weight files (approximately 14.57 GiB)** in the destination folder were compared against the local source with `rclone check`: **34 matching files, 0 differences**. This verifies uploaded content; access without a Google login depends on the folder's sharing settings.

The complete file list and byte sizes are in [weights-manifest.csv](weights-manifest.csv). Preserve the directory names when placing downloaded weights in the repository:

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

| Files | Required for documented evaluation? |
| --- | --- |
| The four initialization files shown in `checkpoints/` | Yes, for each run |
| `final_pascal/foldN.pth` | For Pascal fold N |
| `final_coco/foldN.pth` | For COCO fold N |
| `checkpoints/distilled_*.pth` | No; the default evaluation entry point does not load them separately |
| Other backbone weights / full SAM-H weights | No, when the four initialization files above are available |
| `exp_loss_3thlayer_126/work_dir/ckpt/*.pth` | Training intermediate checkpoints, not needed for evaluation |

You do **not** need to download the entire folder to test one fold. Download the four initialization files and the final checkpoint for that fold. The full folder also includes additional backbones, distilled weights, and intermediate checkpoints for archival purposes.

The default builder loads initialization weights before applying the final checkpoint. `bottleneck_adapter.py` also loads SAM-B when imported, so all four initialization files remain necessary. `my_configs/vitb14_pretrain.yaml` specifies the DINOv2/SAM submodule paths.

Original foundation-model sources: [DINOv2](https://github.com/facebookresearch/dinov2) and [Segment Anything](https://github.com/facebookresearch/segment-anything). The two SAM-H submodule files in Drive are already extracted for the builder. The final loader accepts a raw state dictionary or a dictionary containing `net`, removes `module.` prefixes, and uses strict parameter loading.

## 3. Prepare datasets

`--datapath` is the **parent directory of `VOCdevkit` and/or `COCO2014`**. Use an absolute path. Images and labels are not included in the code or model-weight downloads.

### PASCAL-5i

```text
/path/to/datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        │   ├── 2007_000033.jpg
        │   └── ...
        └── SegmentationClassAug/
            ├── 2007_000033.png
            └── ...
```

Prepare VOC 2012 images and augmented semantic labels. The loader requires `SegmentationClassAug`, rather than only the original VOC `SegmentationClass` folder. PNGs must store semantic IDs: background 0, classes 1–20, ignored pixels 255; RGB visualization masks are not suitable.

Use the provided `splits/pascal/val/fold{0,1,2,3}.txt` files. For the layout above, pass `--datapath /path/to/datasets`; do not append `VOCdevkit` or `VOC2012`.

### COCO-20i

```text
/path/to/datasets/
└── COCO2014/
    ├── train2014/
    │   └── COCO_train2014_*.jpg
    ├── val2014/
    │   └── COCO_val2014_*.jpg
    └── annotations/
        ├── train2014/
        │   └── COCO_train2014_*.png
        └── val2014/
            └── COCO_val2014_*.png
```

Use the supplied `splits/coco/val/fold{0,1,2,3}.pkl` metadata. Each relative image name, such as `val2014/COCO_val2014_*.jpg`, maps to `COCO2014/annotations/val2014/COCO_val2014_*.png`.

The loader requires **preprocessed semantic PNG masks**, with contiguous class IDs 1–80 matching the class order in `dataset/coco.py`, and background 0. Raw COCO annotation JSON alone is insufficient. This release does not provide a standalone JSON-to-semantic-mask conversion script; prepare compatible masks before running COCO evaluation.

## 4. Run evaluation

All commands below use DINOv2-B, 518 × 518 inputs, and one support image per episode.

### Pascal fold 0

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark pascal --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

The script automatically loads `final_pascal/fold0.pth`. Switching `--fold` automatically switches the selected checkpoint.

### COCO fold 0

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark coco --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

This selects `final_coco/fold0.pth`. To override automatic selection, add `--checkpoint /path/to/model.pth`; it must match the chosen benchmark, fold, and model configuration.

### All four folds

```bash
for fold in 0 1 2 3; do
  python test.py \
    --datapath /path/to/datasets \
    --benchmark pascal --fold "$fold" --nshot 1 --bsz 1 \
    --img_size 518 --stop_layer 4 \
    --is_sparse --ce_flag --is_ba --is_pcs
done
```

Replace `--benchmark pascal` with `--benchmark coco` for COCO. Average the four final fold mIoU values to compare with the paper's Mean column. The script's `Mean_mIoU` averages seeds within a single invocation; it is not the four-fold mean.

### GPU selection and output

The release respects `CUDA_VISIBLE_DEVICES`. To use physical GPU 1, prefix any evaluation command with `CUDA_VISIBLE_DEVICES=1`.

Logs are saved under `log_test/<benchmark>/<fold>/_MMDD_HHMMSS.log/log.txt`. The timestamped `.log` component is a directory. Set `--logpath /path/to/logs` to change the base path. Add `--visualize` to save prediction images under `./vis/`.

### Important arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--datapath` | `../` | Dataset parent directory; set explicitly |
| `--benchmark` | `coco` | `pascal` or `coco` |
| `--fold` | `0` | Fold 0–3 |
| `--checkpoint` | Automatic | Optional explicit final checkpoint |
| `--nshot` | `1` | Only 1-shot is accepted in this release |
| `--bsz` | `1` | Use 1 for the documented setup |
| `--img_size` | `518` | This release accepts 518 |
| `--stop_layer` | `4` | Keep 4 for the documented setup |
| `--is_sparse` | False | Enable sparse prompts |
| `--ce_flag` | False | Enable correlation enhancement |
| `--is_ba` | False | Enable the bottleneck adapter path |
| `--is_pcs` | False | Enable the configured correlation path |
| `--is_light` | False | Leave unset for the full model |
| `--nworker` | `0` | Validation currently forces 0 |
| `--use_original_imgsize` | False | Leave unset for resized evaluation |

The four model switches are `store_true` arguments: pass them explicitly as in the commands. They default to False at the CLI.

## 5. Results and evaluation protocol

### Paper results

The following are paper-reported **mIoU (%)**, from Tables I and II, using DINOv2-B. They are not newly measured results from this packaged release.

| Dataset | Setting | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PASCAL-5i | 1-shot | 80.7 | 83.2 | 77.8 | 80.5 | 80.6 |
| PASCAL-5i | 5-shot | 84.4 | 87.3 | 79.0 | 85.5 | 84.1 |
| COCO-20i | 1-shot | 62.2 | 66.0 | 65.5 | 64.2 | 64.5 |
| COCO-20i | 5-shot | 66.6 | 73.3 | 70.9 | 68.1 | 69.7 |

5-shot rows are included for paper reference only. The current model passes `s_input.squeeze(1)` to a four-dimensional image encoder and has no multi-support aggregation in `test.py`; changing the shot count is insufficient to reproduce 5-shot. FSS-1000 is not registered in the dataset factory.

### Recorded local result

The original local evaluation on Pascal fold 0, 1-shot produced:

```text
*** Test [@Epoch 00] Avg L: 0.00000  mIoU: 80.66   FB-IoU: 90.42   ***
```

80.66 rounds to the paper's 80.7 for fold 0. This does not verify the four-fold mean, COCO, or 5-shot results. Tables I and II do not report FB-IoU.

The test entry point uses 1,000 validation episodes, seed array `[0]`, `splits` metadata, ImageNet normalization, and binary predictions `output > 0`. Pascal cycles through fold metadata while sampling supports; COCO samples classes and images.

mIoU averages accumulated foreground IoUs over the fold's classes. FB-IoU pools intersections/unions across the fold separately for background and foreground, then averages those two IoUs. Both are percentages. `Avg L: 0.00000` is a logger placeholder because evaluation supplies `loss=None`; it is not a measured test loss.

### Release checks

- All Python source files parse, and README shell snippets pass syntax checks.
- `python test.py --help` works without loading model weights.
- Model initialization and strict loading of Pascal fold 0 weights succeed on CPU.
- One sample from each of the eight Pascal/COCO validation folds loads with expected image/mask shapes.
- The 34 uploaded model weights match their local source files.

Full GPU evaluation of this release and installation in a new environment remain unverified. The reference package environment passes `pip check`.

## 6. Troubleshooting

| Problem | Action |
| --- | --- |
| Missing Pascal JPG or PNG | Check `--datapath` and the augmented labels directory |
| Missing initialization checkpoint | Download the four required `checkpoints/` files from Drive |
| Missing COCO PNG | Prepare semantic PNG masks; annotation JSON alone is insufficient |
| Missing configuration or split file | Run from the repository root |
| `xFormers not available` | Optional dependency notice; the recorded run completed with it |
| Deprecated `timm.models.layers` warning | Comes from bundled VMamba imports and also appeared in the recorded run |
| `selective_scan_cuda_oflex` import failure | Build/rebuild the extension against the current PyTorch/CUDA stack |
| CUDA unavailable | Check GPU access, driver, CUDA-enabled PyTorch, and `CUDA_VISIBLE_DEVICES` |
| State-dict key mismatch | Check model configuration and checkpoint selection; retain strict loading |
| Unexpected fold results | Check any explicit `--checkpoint` override matches `--fold` |
| Drive requests access | Use an account with access or contact the folder owner; public access depends on sharing settings |

## 7. Citation

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

## 8. Acknowledgements

This repository uses code and models from [DINOv2](https://github.com/facebookresearch/dinov2), [Segment Anything](https://github.com/facebookresearch/segment-anything), and [VMamba](https://github.com/MzeroMiko/VMamba). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/` for attribution and license information.
