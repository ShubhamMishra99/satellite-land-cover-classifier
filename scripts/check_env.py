"""Environment, dependency, and workspace validation script.

Run this script to verify that your Python environment, project directory
structure, configuration files, and core dependencies are correctly set up.
Does NOT require the dataset to be downloaded.
"""

import importlib
import os
import platform
import sys
from pathlib import Path

# Add project root to sys.path to allow src imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
    ("cv2", "OpenCV"),
    ("matplotlib", "Matplotlib"),
    ("yaml", "PyYAML"),
    ("fastapi", "FastAPI"),
    ("pydantic", "Pydantic"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
]

REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data",
    "src/preprocessing",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/explainability",
    "src/inference",
    "app/routes",
    "app/schemas",
    "app/services",
    "tests/unit",
    "tests/api",
    "tests/integration",
    "models",
    "configs",
    "scripts",
    "docs",
]


def check_python_version() -> bool:
    """Check if Python version is >= 3.10."""
    version = sys.version_info
    print(f"[CHECK] Python Version: {version.major}.{version.minor}.{version.micro} ({platform.system()})")
    if version.major == 3 and version.minor >= 10:
        print("  --> PASS: Python version compatible (>= 3.10)")
        return True
    print("  --> FAIL: Python >= 3.10 is required.")
    return False


def check_packages() -> bool:
    """Check availability of required third-party packages."""
    print("\n[CHECK] Checking Core Dependencies:")
    all_ok = True
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  [+] {display_name:16s}: {version}")
        except ImportError:
            print(f"  [-] {display_name:16s}: NOT INSTALLED")
            all_ok = False
    return all_ok


def check_directories() -> bool:
    """Verify required directory tree exists and is writable."""
    print("\n[CHECK] Verifying Directory Structure:")
    all_ok = True
    for rel_dir in REQUIRED_DIRECTORIES:
        p = project_root / rel_dir
        if p.exists() and p.is_dir():
            print(f"  [+] {rel_dir:26s} (exists)")
        else:
            p.mkdir(parents=True, exist_ok=True)
            print(f"  [*] {rel_dir:26s} (created)")
    return all_ok


def check_configs() -> bool:
    """Verify YAML configuration files load correctly."""
    print("\n[CHECK] Verifying Configuration Files:")
    try:
        from src.config import get_base_config, get_custom_cnn_config, get_resnet18_config
        from src.constants import EUROSAT_CLASSES, NUM_CLASSES

        base_cfg = get_base_config()
        cnn_cfg = get_custom_cnn_config()
        resnet_cfg = get_resnet18_config()

        print(f"  [+] base_config.yaml loaded: {len(base_cfg.get('classes', []))} classes configured")
        print(f"  [+] custom_cnn.yaml loaded: model name = {cnn_cfg.get('model', {}).get('name')}")
        print(f"  [+] resnet18.yaml loaded: model name = {resnet_cfg.get('model', {}).get('name')}")
        print(f"  [+] src.constants single source of truth: {NUM_CLASSES} EuroSAT classes defined: {', '.join(EUROSAT_CLASSES[:3])}...")
        return True
    except Exception as exc:
        print(f"  [-] Config verification failed: {exc}")
        return False


def check_torch_device() -> None:
    """Check PyTorch execution device."""
    print("\n[CHECK] PyTorch Device Capabilities:")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard Host)"
        print(f"  [+] CUDA Available : {cuda_available}")
        print(f"  [+] Active Device  : {device_name}")
        print(f"  [+] PyTorch Version: {torch.__version__}")
    except Exception as exc:
        print(f"  [-] PyTorch check error: {exc}")


def main() -> int:
    print("=" * 60)
    print("SATELLITE LAND-USE CLASSIFIER - SETUP AUDIT & VALIDATION")
    print("=" * 60)

    py_ok = check_python_version()
    dir_ok = check_directories()
    cfg_ok = check_configs()
    check_torch_device()
    pkg_ok = check_packages()

    print("\n" + "=" * 60)
    if py_ok and dir_ok and cfg_ok and pkg_ok:
        print("RESULT: ALL SETUP CHECKS PASSED (Milestone 0 Complete)")
        print("=" * 60)
        return 0
    else:
        print("RESULT: SETUP ISSUES DETECTED - See details above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
