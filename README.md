# UINO-FSS

**UINO-FSS: Unifying Representation Learning and Few-Shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation**

Wei Zhuo, Zhiyue Tang, Wufeng Xue, Hao Ding, Junkai Ji, and Linlin Shen
*IEEE Transactions on Image Processing*, vol. 35, 2026.

[Paper](https://doi.org/10.1109/TIP.2026.3723708) · [Model weights — Google Drive](https://drive.google.com/drive/folders/1acBYYjk-UTwcS798ueKlb-rXPQtnCIy_)

UINO-FSS combines a DINOv2 encoder with a lightweight segmenter. Hierarchical distillation aligns early DINOv2 representations with SAM embeddings through a bottleneck adapter. Semantic-aware visual prompts and Mamba-based hypercorrelation guide few-shot segmentation.

This release documents **1-shot evaluation on PASCAL-5i and COCO-20i**. Training instructions are not included. The paper also reports 5-shot and FSS-1000 results; those evaluation paths are not supported by this release.

## 1. Build the evaluation environment

### Tested platform and prerequisites

**Validated in a fresh Conda environment on November 5, 2026:** Pascal fold 0, 1-shot completed all 1,000 episodes with **80.66 mIoU / 90.42 FB-IoU**. COCO fold 0 also passed a real-image GPU smoke test. See the [validation record](environment/validation-2026-11-05.md).

Use **Linux x86_64**. The validation platform is Ubuntu 22.04, an NVIDIA RTX A6000 (compute capability 8.6), and NVIDIA driver 560.35.03. Install Conda and Git first, and make sure `nvidia-smi` works in the same terminal/container that will run the commands. The host GPU driver is not installed by Conda.

This CUDA 11.8 / PyTorch 2.3.1 recipe targets the tested hardware/software baseline. It does not claim support for every GPU generation, Windows, macOS, ARM, or a container without GPU access. Keep at least 30 GB free for the environment, package caches, toolkit, and build; weights and datasets need additional space. Network access to the Conda channels, PyPI, the PyTorch wheel index, and GitHub is required.

The installation uses:

| Component | Version |
| --- | --- |
| Python | 3.10.22 |
| PyTorch / torchvision | 2.3.1+cu118 / 0.18.1+cu118 |
| CUDA development components | 11.8 (`nvcc` 11.8.89) |
| Conda GCC / G++ | 11.2.0 |
| NumPy | 2.2.6 |
| Albumentations / OpenCV headless | 2.0.8 / 4.11.0.86 |
| Triton | 2.3.1 |
| Mamba CUDA backend | mamba-ssm 2.2.2 |
| VMamba selective-scan source | VMamba commit `2ed52ead062a51a64521ed3871d52914bf532876` |

[`environment/conda-linux-64.lock`](environment/conda-linux-64.lock) records exact Conda artifacts and checksums for Python, CUDA headers/libraries, and the compiler. [`requirements.txt`](requirements.txt) pins the evaluation Python packages. [`environment/dinov2-packages.txt`](environment/dinov2-packages.txt) is the older complete `dinov2` reference snapshot, including unrelated packages; do not install that snapshot as a requirements file.

### Step 1 — clone and create a new environment

Run these commands in Bash. Use a new environment; do not install into `base` or overwrite an existing experiment environment.

```bash
git clone https://github.com/SZUCV1009/UINO-FSS.git
cd UINO-FSS

conda create -n uino-fss --file environment/conda-linux-64.lock -y
conda activate uino-fss

python --version
nvidia-smi
```

The Conda lock includes **CUDA 11.8 nvcc, the runtime and development headers/libraries required by selective-scan, and GCC/G++**. Profiling GUI tools are not required. No system CUDA installation or `sudo` is needed for the environment itself. The host NVIDIA driver and GPU access remain prerequisites. The explicit lock avoids re-solving toolkit/compiler versions against changing channel metadata.

### Step 2 — select this environment's CUDA and compiler

```bash
source scripts/activate_cuda.sh

"$CUDA_HOME/bin/nvcc" --version
"$CXX" --version
```

This sets `CUDA_HOME`, `CC`, `CXX`, `CUDAHOSTCXX`, the library path, and `MAX_JOBS=2` to the active Conda environment. It prevents accidentally compiling with a system CUDA 12.x toolkit. Source this file again after opening a new terminal and activating `uino-fss`. Do not add CUDA's `lib/stubs` directory to `LD_LIBRARY_PATH`.

### Step 3 — install the pinned Python dependencies

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip check
```

Install the official CUDA 11.8 torch/torchvision wheels **before** the remaining requirements. The second command pins their dependencies to the tested versions. Do not subsequently upgrade torch, torchvision, NumPy, Triton, or the CUDA build stack independently.

DINOv2 and SAM source modules are bundled in this repository. xFormers is optional. The native Mamba backend is required by the model's `v0` forward path and is installed in the next step; a VMamba `oflex` installation alone is insufficient.

### Step 4 — install both CUDA scan backends

Install Mamba's native `selective_scan_cuda` backend first:

```bash
python -m pip install --no-build-isolation --no-deps mamba-ssm==2.2.2 \
  --index-url https://pypi.org/simple
```

Its installer uses a compatible upstream wheel when available, otherwise builds locally. The compiler and CUDA headers from Step 2 must remain available. Its Python dependencies, including transformers, are pinned in `requirements.txt`.

Next build VMamba's `selective_scan_cuda_oflex` backend. Keep an NVIDIA GPU visible during the build: upstream setup detects the visible GPU's compute capability. This is a native build for that architecture; rebuild if moving to a different GPU architecture or changing PyTorch/CUDA.

```bash
mkdir -p .build
git clone https://github.com/MzeroMiko/VMamba.git .build/VMamba
git -C .build/VMamba checkout --detach 2ed52ead062a51a64521ed3871d52914bf532876

python -m pip install --no-build-isolation --no-deps \
  ./.build/VMamba/kernels/selective_scan
python -m pip check
```

The commit is fixed, so the build does not silently change with upstream HEAD. `--no-build-isolation` uses the installed PyTorch/CUDA toolchain; `--no-deps` prevents the extension installer from replacing the pinned packages. This step can take several minutes. If rerunning it, reuse the existing `.build/VMamba` checkout instead of cloning into a nonempty directory.

### Step 5 — verify actual GPU execution

```bash
python scripts/check_environment.py
python test.py --help
```

The check verifies all pinned package versions, the NumPy↔Tensor bridge, GPU matrix multiplication, torchvision CUDA NMS, and both Mamba and VMamba CUDA scan kernels against the PyTorch reference implementation. It must finish with `Environment checks passed.`; an import alone is not a sufficient installation test.

After downloading the required weights and preparing a dataset using Sections 2–3, check a complete real-image prediction:

```bash
python scripts/check_environment.py \
  --datapath /path/to/datasets --benchmark pascal --fold 0
```

This strictly loads the final checkpoint, reads an episode, performs GPU segmentation, and computes intersection/union. It checks that the inference pipeline runs; it is not a full mIoU evaluation. Use Section 4 for the complete benchmark.

### Installation troubleshooting

- **CUDA unavailable:** check `nvidia-smi`, driver installation, GPU visibility, and container GPU passthrough. Conda cannot fix missing host GPU access.
- **CUDA version mismatch / missing headers:** activate `uino-fss` and source `scripts/activate_cuda.sh` before building. `CUDA_HOME` must point to the active environment.
- **Compiler mismatch:** use the locked Conda GCC/G++ 11.2, not an arbitrary system compiler.
- **Missing `selective_scan_cuda` / `NameError` in `forwardv0`:** install `mamba-ssm==2.2.2` as in Step 4; `oflex` alone does not supply that module.
- **Extension import or `no kernel image` error:** rebuild the extension on the target GPU architecture with this exact torch/CUDA stack.
- **Build killed for memory pressure:** retry with `export MAX_JOBS=1` before the extension installation command.
- **Download or TLS failure:** check the machine's network/proxy and certificates. These commands require reachable package servers; do not disable certificate verification.

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

### Fresh-environment reproduction

On November 5, 2026, the new `uino-fss` environment completed the packaged `test.py` on Pascal fold 0, 1-shot, seed 0, for all 1,000 episodes. It reproduced the earlier `dinov2` environment result:

```text
*** Test [@Epoch 00] Avg L: 0.00000  mIoU: 80.66   FB-IoU: 90.42   ***
```

80.66 rounds to the paper's 80.7 for fold 0. This does not verify the four-fold mean, COCO, or 5-shot results. Tables I and II do not report FB-IoU.

The test entry point uses 1,000 validation episodes, seed array `[0]`, `splits` metadata, ImageNet normalization, and binary predictions `output > 0`. Pascal cycles through fold metadata while sampling supports; COCO samples classes and images.

mIoU averages accumulated foreground IoUs over the fold's classes. FB-IoU pools intersections/unions across the fold separately for background and foreground, then averages those two IoUs. Both are percentages. `Avg L: 0.00000` is a logger placeholder because evaluation supplies `loss=None`; it is not a measured test loss.

### Release checks

- All Python source files parse, and README shell snippets pass syntax checks.
- `python test.py --help` works without loading model weights.
- Strict loading of Pascal/COCO fold 0 weights and real-image GPU inference succeed in the fresh environment.
- One sample from each of the eight Pascal/COCO validation folds loads with expected image/mask shapes.
- The 34 uploaded model weights match their local source files.

A fresh `uino-fss` environment has passed `pip check`, both CUDA scan kernel comparisons, real-image GPU smoke tests for Pascal and COCO fold 0, and the full 1,000-episode Pascal fold 0 evaluation. The Conda lock has also been used to recreate a separate clean environment. See Section 1 for the tested platform and installation commands.

## 6. Troubleshooting

| Problem | Action |
| --- | --- |
| Missing Pascal JPG or PNG | Check `--datapath` and the augmented labels directory |
| Missing initialization checkpoint | Download the four required `checkpoints/` files from Drive |
| Missing COCO PNG | Prepare semantic PNG masks; annotation JSON alone is insufficient |
| Missing configuration or split file | Run from the repository root |
| `xFormers not available` | Optional dependency notice; the recorded run completed with it |
| Deprecated `timm.models.layers` warning | Comes from bundled VMamba imports and also appeared in the recorded run |
| `selective_scan_cuda` import failure / `NameError` | Install `mamba-ssm==2.2.2` using Section 1; the model requires this backend |
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
