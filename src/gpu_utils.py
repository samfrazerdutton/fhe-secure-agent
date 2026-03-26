import cupy as cp
from pathlib import Path

def get_device_info():
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    major = props['major']
    minor = props['minor']
    sm = f"sm_{major}{minor}"
    name = props['name'].decode()
    vram_gb = props['totalGlobalMem'] / (1024**3)
    return {'name': name, 'sm': sm, 'major': major, 'minor': minor, 'vram_gb': vram_gb}

def get_ptx(kernels_dir: Path, kernel_name: str) -> Path:
    info = get_device_info()
    sm = info['sm']
    
    # Try exact match first, then fall back to closest lower SM
    ptx = kernels_dir / f"{kernel_name}_{sm}.ptx"
    if ptx.exists():
        return ptx
    
    # Fallback: find highest SM that is <= current GPU
    major, minor = info['major'], info['minor']
    sm_int = major * 10 + minor
    
    candidates = sorted(kernels_dir.glob(f"{kernel_name}_sm_*.ptx"))
    best = None
    for c in candidates:
        try:
            c_sm = int(c.stem.split('_sm_')[1])
            if c_sm <= sm_int:
                best = c
        except:
            continue
    
    if best:
        print(f"[gpu_utils] No PTX for {sm}, using {best.name}")
        return best
    
    raise RuntimeError(f"No compatible PTX found for {sm} in {kernels_dir}")
