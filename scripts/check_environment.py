"""Verify native CUDA dependencies; optionally run one real segmentation episode."""
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", help="Also test a real dataset episode and final checkpoint")
    parser.add_argument("--benchmark", choices=["pascal", "coco"], default="pascal")
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The documented installation targets Linux x86_64.")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Activate the Python 3.10 uino-fss environment first.")

    import numpy as np
    import torch
    import torchvision
    from torchvision.ops import nms
    from packaging.requirements import Requirement

    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        version = importlib.metadata.version(requirement.name)
        if not requirement.specifier.contains(version):
            raise RuntimeError(f"Version mismatch: {requirement}; installed {version}")
    print(f"Python {platform.python_version()}, torch {torch.__version__}, torchvision {torchvision.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the host NVIDIA driver and GPU access.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA runtime: {torch.version.cuda}")
    array = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal(torch.from_numpy(array).cuda().cpu().numpy(), array)
    matrix = torch.eye(8, device="cuda")
    torch.testing.assert_close(matrix @ matrix, matrix)
    boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]], device="cuda", dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], device="cuda")
    assert nms(boxes, scores, 0.5).tolist() == [0]
    print("NumPy bridge, CUDA matrix multiplication, and torchvision CUDA NMS: OK")

    # Load only the kernel wrapper: importing the model package also requires weights.
    import selective_scan_cuda  # noqa: F401
    import selective_scan_cuda_oflex  # noqa: F401
    print(f"Mamba {importlib.metadata.version('mamba-ssm')}, "
          f"VMamba selective_scan {importlib.metadata.version('selective_scan')}")
    path = ROOT / "segment_anything_dinov2/modeling/vmamba_models/csms6s.py"
    spec = importlib.util.spec_from_file_location("uino_scan_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    u = torch.randn(1, 16, 32, device="cuda")
    delta = torch.randn_like(u)
    a = -torch.rand(16, 8, device="cuda")
    b = torch.randn(1, 4, 8, 32, device="cuda")
    c = torch.randn_like(b)
    d = torch.ones(16, device="cuda")
    bias = torch.zeros_like(d)
    with torch.no_grad():
        expected = module.selective_scan_fn(u, delta, a, b, c, d, bias, True, backend="torch")
        for backend in ("mamba", "oflex"):
            actual = module.selective_scan_fn(u, delta, a, b, c, d, bias, True, backend=backend)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
            print(f"Selective-scan {backend} CUDA kernel vs. PyTorch reference: OK")
    torch.cuda.synchronize()

    if args.datapath:
        # Configuration and split files currently use repository-relative paths.
        os.chdir(ROOT)
        from omegaconf import OmegaConf
        from common import utils
        from common.evaluation import Evaluator
        from dataset.dataset import FSSDataset
        from segment_anything_dinov2.build_fs_dino import _build_fs_dino

        utils.fix_randseed(0)
        cfg = OmegaConf.load(ROOT / "my_configs/vitb14_pretrain.yaml")
        model = _build_fs_dino(cfg)
        checkpoint = args.checkpoint or ROOT / f"final_{args.benchmark}/fold{args.fold}.pth"
        state = torch.load(checkpoint, map_location="cpu")
        state = state["net"] if "net" in state else state
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
        model.cuda().eval()
        FSSDataset.initialize(518, args.datapath, False)
        loader, _, _ = FSSDataset.build_dataloader(args.benchmark, 1, 0, args.fold, "val", 1, "splits")
        batch = utils.to_cuda(next(iter(loader)))
        with torch.no_grad():
            _, output = model(batch["query_img"], batch["support_imgs"], batch["support_masks"],
                              4, False, True, True, True, True)
        prediction = torch.where(output > 0, 1, 0).squeeze(1)
        assert prediction.shape == batch["query_mask"].shape
        assert torch.isfinite(output).all().item()
        Evaluator.initialize()
        intersection, union = Evaluator.classify_prediction(prediction.clone(), batch)
        torch.cuda.synchronize()
        print(f"{args.benchmark} fold {args.fold}: strict weights, data loading, GPU prediction, and metric computation: OK")
        print(f"Prediction shape: {tuple(prediction.shape)}; foreground intersection/union: "
              f"{intersection[1].sum().item()}/{union[1].sum().item()}")
        print("One-episode smoke test only; this is not a benchmark mIoU measurement.")
    print("Environment checks passed.")


if __name__ == "__main__":
    main()
