import subprocess
import cupy as cp
from pathlib import Path

def get_device_info():
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    major = props['major']
    minor = props['minor']
    sm = f"sm_{major}{minor}"
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    vram_gb = props['totalGlobalMem'] / (1024**3)
    return {'name': name, 'sm': sm, 'vram_gb': vram_gb}

def get_ptx(kernels_dir: Path, kernel_name: str) -> Path:
    info = get_device_info()
    sm = info['sm']
    ptx_path = kernels_dir / f"{kernel_name}_{sm}.ptx"

    if not ptx_path.exists():
        print(f"[gpu_utils] No PTX for {sm}, compiling...")
        cu_path = kernels_dir / f"{kernel_name}.cu"
        if not cu_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {cu_path}")
        result = subprocess.run(
            ["nvcc", f"--gpu-architecture={sm}", "--ptx",
             str(cu_path), "-o", str(ptx_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")
        print(f"[gpu_utils] Compiled {ptx_path.name}")

    return ptx_path
