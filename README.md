# UINO-FSS

**UINO-FSS: Unifying Representation Learning and Few-Shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation**
Wei Zhuo, Zhiyue Tang, Wufeng Xue, Hao Ding, Junkai Ji, and Linlin Shen
*IEEE Transactions on Image Processing*, vol. 35, 2026.
[Paper](https://doi.org/10.1109/TIP.2026.3723708)

UINO-FSS combines a DINOv2 encoder with a lightweight segmenter. Hierarchical distillation transfers SAM's embedding knowledge into a bottleneck adapter. A meta-visual prompt generator uses semantic-aware visual prompts and Mamba-based hypercorrelation to guide segmentation.

This README covers **evaluation only**, based on the current repository. Training and distillation instructions are outside its scope.

## Contents

- [Evaluation status](#evaluation-status)
- [Environment setup](#environment-setup)
- [Repository layout](#repository-layout)
- [Model checkpoints](#model-checkpoints)
- [Dataset preparation](#dataset-preparation)
- [Evaluation](#evaluation)
- [Results](#results)
- [Arguments and evaluation protocol](#arguments-and-evaluation-protocol)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Evaluation status

| Setting | Status in the current code |
| --- | --- |
| PASCAL-5i, fold 0, 1-shot | Recorded run: **80.66 mIoU / 90.42 FB-IoU**; agrees with the paper's rounded fold 0 mIoU of 80.7 |
| PASCAL-5i, folds 1–3, 1-shot | Loader and checkpoint files are present locally; matching weights are selected automatically; results have not been independently verified for this README |
| COCO-20i, 1-shot | Loader and checkpoint files are present locally; matching weights are selected automatically; results have not been independently verified for this README |
| 5-shot | Reported in the paper, but **not directly supported by the current model forward path** |
| FSS-1000 | Reported in the paper, but not registered in the current dataset factory |

The evaluation entry point automatically selects the checkpoint for the requested benchmark and fold. Use `--checkpoint` to override it. GPU selection follows `CUDA_VISIBLE_DEVICES`. Local checkpoint availability does not imply public download availability.

## Environment setup

### Requirements

- Linux and an NVIDIA GPU with a working CUDA driver.
- Python 3.10.
- PyTorch 2.3.1 and torchvision 0.18.1, with CUDA 11.8 wheels.
- A CUDA toolkit with `nvcc` and a C++ compiler if building the selective-scan extension.

The evaluation utilities call `.cuda()` directly, so CPU-only evaluation is not supported by the current code. GPU memory requirements have not been systematically measured for this release.

### Create an environment

Clone the repository, then run commands from its root:

```bash
git clone https://github.com/SZUCV1009/UINO-FSS.git
cd UINO-FSS
```

Create the environment:

```bash
conda create -n uino-fss python=3.10 -y
conda activate uino-fss

python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements.txt
```

These versions follow the relevant installed packages in the local `dinov2` environment (Python 3.10.18, torch 2.3.1+cu118, torchvision 0.18.1+cu118), except that NumPy is pinned to 1.26.4 for compatibility with older compiled dependencies. The local environment has NumPy 2.2.6. **This clean-environment recipe has not yet been validated end to end** and is not a fully locked environment export. Matching CUDA wheel commands are documented in the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/).

The repository includes the DINOv2 and SAM source modules used here. Installing separate copies is unnecessary for this evaluation entry point. Run from the repository root to resolve local imports, configuration files, and split metadata.

### Selective-scan CUDA extension

The local environment includes VMamba's `selective_scan==0.0.2`. Build the `selective_scan_cuda_oflex` backend from the [upstream VMamba kernel source](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan):

```bash
nvcc --version
git clone https://github.com/MzeroMiko/VMamba.git /tmp/uino-fss-vmamba
python -m pip install --no-build-isolation /tmp/uino-fss-vmamba/kernels/selective_scan
python -c "import selective_scan_cuda_oflex; print('Selective scan CUDA extension is available')"
```

Build with an available NVIDIA GPU and a toolkit/compiler compatible with the installed PyTorch build. Rebuild after changing PyTorch or CUDA. This command tracks current upstream source, rather than a verified VMamba release commit for this repository.

The bundled selective-scan implementation falls back to PyTorch when no CUDA scan backend can be imported; evaluation may then be considerably slower. PyTorch's Linux CUDA installation also supplies Triton for cross-scan kernels. The local environment contains `mamba-ssm` and `causal-conv1d`, but this evaluation path does not directly import them, and they are not required when using the VMamba extension above.

`xformers` is optional for this configuration. The recorded Pascal run completed without it.

### Check CUDA

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available())"
```

## Repository layout

```text
uino-fss/
├── test.py                         # Evaluation entry point
├── my_configs/
│   └── vitb14_pretrain.yaml         # DINOv2-B / SAM initialization
├── checkpoints/                    # Foundation-model initialization weights
├── final_pascal/                   # Pascal fold0.pth ... fold3.pth
├── final_coco/                     # COCO fold0.pth ... fold3.pth
├── dataset/                        # Dataset loaders and transforms
├── splits/
│   ├── pascal/val/fold0.txt         # Pascal evaluation metadata
│   └── coco/val/fold0.pkl           # COCO class-to-image metadata
├── common/                         # Metrics, logging, visualization
├── dinov2/                         # Bundled DINOv2 modules
├── segment_anything/                # Bundled SAM modules
└── segment_anything_dinov2/         # UINO-FSS model and VMamba modules
```

Keep all four validation fold files for each benchmark. `test.py` uses the `splits` loader; `lists/` is not its evaluation metadata source.

## Model checkpoints

### Required files

```text
checkpoints/
├── dinov2_vitb14_pretrain.pth
├── sam_vit_b_01ec64.pth
├── sam_maskDecoder_H.pth
└── sam_promptEncoder_H.pth

final_pascal/
├── fold0.pth
├── fold1.pth
├── fold2.pth
└── fold3.pth

final_coco/
├── fold0.pth
├── fold1.pth
├── fold2.pth
└── fold3.pth
```

An individual run needs the target benchmark/fold's final checkpoint and the four initialization files.

| File | Purpose / source |
| --- | --- |
| `dinov2_vitb14_pretrain.pth` | Non-register DINOv2 ViT-B/14 weights from [DINOv2](https://github.com/facebookresearch/dinov2) |
| `sam_vit_b_01ec64.pth` | Official SAM ViT-B weights from [Segment Anything](https://github.com/facebookresearch/segment-anything); loaded at import time by `bottleneck_adapter.py` |
| `sam_maskDecoder_H.pth` | Mask decoder state dictionary extracted from SAM ViT-H |
| `sam_promptEncoder_H.pth` | Prompt encoder state dictionary extracted from SAM ViT-H |
| `final_pascal/fold*.pth` | UINO-FSS Pascal weights; **public download URL to be added** |
| `final_coco/fold*.pth` | UINO-FSS COCO weights; **public download URL to be added** |

The builder loads initialization weights **before** the final UINO-FSS checkpoint, so initialization files are still required. Separate `distilled_*.pth` files are not loaded by the default test path.

If the extracted SAM-H files are unavailable, obtain `sam_vit_h_4b8939.pth` from the official SAM model links, put it in `checkpoints/`, and extract the submodule state dictionaries:

```bash
python - <<'PY'
from pathlib import Path
import torch

root = Path('checkpoints')
state = torch.load(root / 'sam_vit_h_4b8939.pth', map_location='cpu')
for prefix, filename in [
    ('mask_decoder.', 'sam_maskDecoder_H.pth'),
    ('prompt_encoder.', 'sam_promptEncoder_H.pth'),
]:
    substate = {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix)}
    if not substate:
        raise RuntimeError(f'No keys found for {prefix}')
    torch.save(substate, root / filename)
    print(f'Saved {filename}: {len(substate)} entries')
PY
```

The full SAM-H file is only needed for extraction. The final checkpoint loader accepts a raw state dictionary or a dictionary containing `net`, removes the `module.` prefix, and loads parameters with `strict=True`.

## Dataset preparation

`--datapath` is the **parent directory** containing `VOCdevkit` and/or `COCO2014`. Use an absolute path.

### PASCAL-5i

Prepare VOC 2012 images and augmented semantic segmentation labels:

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

The loader requires `SegmentationClassAug`, rather than only the original VOC `SegmentationClass` folder. PNG masks must contain semantic IDs: background 0, classes 1–20, and ignored pixels 255. RGB visualization masks are not a substitute. Preserve `splits/pascal/val/fold{0,1,2,3}.txt` for comparisons.

For this layout, pass `--datapath /path/to/datasets`. Do not append `VOCdevkit` or `VOC2012`.

### COCO-20i

Prepare COCO 2014 images and **preprocessed semantic PNG masks**:

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

`splits/coco/val/fold*.pkl` supplies relative names such as `val2014/COCO_val2014_*.jpg`. For each name, the loader expects `COCO2014/annotations/<relative-name-with-.png>`.

Masks must use contiguous semantic class IDs 1–80 matching `dataset/coco.py`, with 0 for background. Raw COCO instance annotation JSON files alone are insufficient. A standalone JSON-to-semantic-mask preparation script is not included; obtain masks compatible with the supplied splits or prepare them before evaluation.

## Evaluation

### 1. Select the checkpoint

By default, `test.py` selects `final_<benchmark>/fold<fold>.pth`, for example `final_pascal/fold0.pth` for Pascal fold 0. No source edit is needed when switching benchmarks or folds.

To use a different location, add `--checkpoint /path/to/model.pth` to the command. The checkpoint must still match the selected benchmark, fold, and DINOv2-B model configuration.

### 2. PASCAL-5i, 1-shot

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark pascal --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

### 3. COCO-20i, 1-shot

This command automatically selects `final_coco/fold0.pth`:

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark coco --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

### 4. Evaluate all four folds

Each invocation automatically loads the matching fold checkpoint:

```bash
for fold in 0 1 2 3; do
  python test.py \
    --datapath /path/to/datasets \
    --benchmark pascal --fold "$fold" --nshot 1 --bsz 1 \
    --img_size 518 --stop_layer 4 \
    --is_sparse --ce_flag --is_ba --is_pcs
done
```

For COCO, change `--benchmark pascal` to `--benchmark coco`. Compute the arithmetic mean of the four **final fold mIoU values**. The script's `Mean_mIoU` is the seed mean within one invocation, not a four-fold mean.

### 5. GPU selection, logs, and visualization

The script respects `CUDA_VISIBLE_DEVICES`. To evaluate on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py \
  --datapath /path/to/datasets \
  --benchmark pascal --fold 0 --nshot 1 --bsz 1 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

Logs are saved to:

```text
log_test/<benchmark>/<fold>/_MMDD_HHMMSS.log/log.txt
```

`--logpath /path/to/logs` changes the base directory. The timestamped `.log` component is a directory. Add `--visualize` to a single-fold command to save prediction images under `./vis/`.

### 5-shot and FSS-1000 limitations

The CLI restricts `--nshot` to 1 because the model calls `s_input.squeeze(1)` before passing support images to a four-dimensional image encoder. This only removes the shot dimension for `nshot=1`; `test.py` implements no multi-support prediction aggregation. The Pascal loader also converts the support-image list into one NumPy array, which can fail for differently sized supports. Simply setting `--nshot 5` is therefore not a valid 5-shot reproduction procedure for this snapshot.

`--benchmark` is restricted to `pascal` and `coco`. FSS-1000 needs an additional loader and the appropriate evaluation/checkpoint setup.

## Results

The following are **paper-reported mIoU (%)**, not newly measured results. Backbone: DINOv2-B. Source: Tables I and II.

### PASCAL-5i

| Setting | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-shot | 80.7 | 83.2 | 77.8 | 80.5 | 80.6 |
| 5-shot | 84.4 | 87.3 | 79.0 | 85.5 | 84.1 |

### COCO-20i

| Setting | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-shot | 62.2 | 66.0 | 65.5 | 64.2 | 64.5 |
| 5-shot | 66.6 | 73.3 | 70.9 | 68.1 | 69.7 |

### Available reproduction record

The Pascal fold 0, 1-shot run reported during local evaluation produced:

```text
*** Test [@Epoch 00] Avg L: 0.00000  mIoU: 80.66   FB-IoU: 90.42   ***
```

80.66 rounds to the paper's 80.7 for this fold. This run does not establish the four-fold mean or COCO results. Tables I and II do not report FB-IoU, so 90.42 cannot be directly checked against them.

## Arguments and evaluation protocol

| Argument | Current default | Notes |
| --- | --- | --- |
| `--datapath` | `../` | Parent of `VOCdevkit` and/or `COCO2014`; set explicitly |
| `--benchmark` | `coco` | `pascal` or `coco`; automatically selects matching weights |
| `--checkpoint` | Automatic | Optional explicit evaluation checkpoint path |
| `--fold` | `0` | 0–3; must match the checkpoint |
| `--nshot` | `1` | Use 1 for the current forward path |
| `--bsz` | `1` | Use 1 for the documented configuration |
| `--nworker` | `0` | Validation loader forces 0 regardless of this argument |
| `--img_size` | `518` | Keep 518 to match config/model assumptions |
| `--stop_layer` | `4` | Keep 4 for the documented configuration |
| `--is_sparse` | `False` | Pass explicitly for sparse prompts |
| `--ce_flag` | `False` | Pass explicitly for correlation enhancement |
| `--is_ba` | `False` | Pass explicitly for the bottleneck adapter path |
| `--is_pcs` | `False` | Pass explicitly for the configured correlation path |
| `--is_light` | `False` | Leave unset for the full path |
| `--logpath` | `./log_test` | Base output directory |
| `--visualize` | `False` | Save images to `./vis/` |
| `--use_original_imgsize` | `False` | Leave unset for resized evaluation |

Boolean switches use `action='store_true'`: their actual CLI defaults are **False**, even where comments or model-method defaults suggest otherwise. The commands explicitly enable the four required switches.


The current evaluation protocol is:

- DINOv2 ViT-B/14 from `my_configs/vitb14_pretrain.yaml`; inputs resized to 518 × 518 and ImageNet-normalized.
- `split_type='splits'`, no dataloader shuffling, 1,000 episodes per validation run, and seed array `[0]`.
- Pascal cycles through fold metadata and samples supports; COCO samples classes and images to construct episodes.
- Binary predictions use `output > 0`.
- mIoU averages foreground IoUs across the fold's classes, after accumulating intersections/unions per class. FB-IoU pools intersections/unions over the fold separately for foreground and background, then averages those two IoUs. Both are percentages.
- `Avg L: 0.00000` is expected: the test loop supplies `loss=None`, and the logger records zero; this is not a measured loss.

Keep splits, checkpoints, seed, flags, image size, and dependency versions consistent for comparisons. The single-seed protocol describes this code snapshot; it does not establish an otherwise unspecified paper seed-averaging protocol.

## Troubleshooting

| Symptom | Check / action |
| --- | --- |
| Missing `../VOCdevkit/VOC2012/JPEGImages/...jpg` | Set `--datapath` to the parent of `VOCdevkit`; check JPG and augmented PNG availability |
| Missing `sam_vit_b_01ec64.pth` during model import | Adapter imports construct SAM; put official SAM-B weights in `checkpoints/` |
| Missing `sam_maskDecoder_H.pth` / `sam_promptEncoder_H.pth` | Extract the SAM-H submodule state dictionaries as above |
| Missing COCO PNG masks | Provide semantic masks; raw annotation JSON is insufficient |
| Missing config or split files | Run from the repository root |
| `xFormers not available` | Optional dependency notice; the recorded Pascal run completed with it |
| Deprecated `timm.models.layers` import warning | Comes from bundled VMamba imports; also printed in the recorded run |
| Cannot import `selective_scan_cuda_oflex` | Build/rebuild the VMamba extension for the current torch/CUDA; fallback can be slower |
| CUDA unavailable / CUDA tensor errors | Check CUDA-enabled torch, GPU driver, and `CUDA_VISIBLE_DEVICES` |
| Missing/unexpected state-dict keys | Match model config and UINO-FSS weights; retain strict loading to expose mismatches |
| Unexpected results on a different fold | Default selection follows `--fold`; check any explicit `--checkpoint` override |
| Dimension errors with `--nshot 5` | CLI accepts only 1-shot; the current forward path assumes one support image |

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

This repository uses code and models from [DINOv2](https://github.com/facebookresearch/dinov2), [Segment Anything](https://github.com/facebookresearch/segment-anything), and [VMamba](https://github.com/MzeroMiko/VMamba). Retain their copyright and license notices and refer to the upstream projects for their respective terms.

## Release validation

The evaluation-only package was checked in the existing local environment:

- All Python source files parse successfully, and README shell snippets pass shell syntax checks.
- `python test.py --help` works without loading model weights.
- Model initialization and strict loading of the Pascal fold 0 final checkpoint succeed on CPU.
- One sample from each of the four Pascal and four COCO folds loads with the expected query/support image and mask shapes.

GPU inference and installation into a new environment were not repeated for this release check. The Pascal metric above is the previously recorded run, not a new measurement of this packaged version.
