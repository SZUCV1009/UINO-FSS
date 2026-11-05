# Fresh-environment validation — November 5, 2026

## Platform

- Ubuntu 22.04.5 LTS, Linux x86_64.
- NVIDIA RTX A6000, compute capability 8.6; driver 560.35.03.
- Separate Conda environment named `uino-fss`; the original `dinov2` environment was not modified.
- Python 3.10.22; torch 2.3.1+cu118; torchvision 0.18.1+cu118; NumPy 2.2.6.
- Conda CUDA 11.8 compiler/runtime/development components; nvcc 11.8.89; GCC/G++ 11.2.0.
- Mamba 2.2.2 native `selective_scan_cuda` backend.
- VMamba `selective_scan` 0.0.2, compiled from commit `2ed52ead062a51a64521ed3871d52914bf532876` using the Conda toolchain.

## Installation checks

1. Created an empty Conda environment, installed official PyTorch/torchvision CUDA 11.8 wheels, and installed the pinned evaluation requirements.
2. Exported exact Conda artifacts with MD5 checksums and recreated a second empty environment from `conda-linux-64.lock` using the package cache. The lock contains 56 Conda artifacts.
3. Compiled and installed VMamba's `oflex` extension using the environment's GCC/G++ and CUDA headers/libraries. Mamba 2.2.2's installer obtained its matching upstream cu118 / torch2.3 / cp310 / cxx11abiFALSE release wheel.
4. `python -m pip check`: **No broken requirements found.**
5. Validated the locked Python package versions and executed the checks below on the GPU.

The old README omitted a required backend: the model's `v0` path explicitly calls `selective_scan_cuda` from Mamba. Installing only VMamba's `oflex` extension did not run the complete model. The revised instructions install and check both backends.

## Native CUDA and complete-model checks

`python scripts/check_environment.py --datapath /path/to/datasets --benchmark pascal --fold 0`:

```text
Python 3.10.22, torch 2.3.1+cu118, torchvision 0.18.1+cu118
GPU: NVIDIA RTX A6000; CUDA runtime: 11.8
NumPy bridge, CUDA matrix multiplication, and torchvision CUDA NMS: OK
Mamba 2.2.2, VMamba selective_scan 0.0.2
Selective-scan mamba CUDA kernel vs. PyTorch reference: OK
Selective-scan oflex CUDA kernel vs. PyTorch reference: OK
pascal fold 0: strict weights, data loading, GPU prediction, and metric computation: OK
Prediction shape: (1, 518, 518); foreground intersection/union: 42038/45469.0
Environment checks passed.
```

The corresponding COCO fold 0 check also passed strict checkpoint loading, real data loading, GPU prediction, and metric computation. These checks each use one episode; they do not measure dataset-wide mIoU.

The displayed `/path/to/datasets` is a portable placeholder for the dataset root used during validation. The optional xFormers notice and the timm deprecated-import warning occurred without preventing evaluation.

## Full Pascal evaluation

Executed the packaged `test.py` in the fresh environment with:

```bash
python test.py \
  --datapath /path/to/datasets \
  --benchmark pascal --fold 0 --nshot 1 --bsz 1 \
  --img_size 518 --stop_layer 4 \
  --is_sparse --ce_flag --is_ba --is_pcs
```

Completed **1,000/1,000 episodes**, seed 0, with exit code 0:

```text
[Batch: 1000/1000] mIoU: 80.66  |  FB-IoU: 90.42
*** Test [@Epoch 00] Avg L: 0.00000  mIoU: 80.66   FB-IoU: 90.42   ***
Fold 0 mIoU: 80.66   FB-IoU: 90.42
```

This matches the earlier Pascal fold 0 result from `dinov2`. Other folds, a full COCO benchmark, other GPU architectures, and other operating systems were not evaluated in this environment validation.
