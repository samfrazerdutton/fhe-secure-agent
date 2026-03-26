import numpy as np
from .fhe_bridge import N, T

def encode(text: str) -> np.ndarray:
    """Text -> integer vector safe for BFV (values < T)"""
    # Store each character as two slots: high nibble + low nibble
    # ord('A')=65 -> [4, 1], recoverable exactly
    vec = np.zeros(N, dtype=np.uint32)
    chars = text[:N//2]  # max 512 chars
    for i, ch in enumerate(chars):
        code = ord(ch) & 0xFF
        vec[i*2]   = (code >> 4) & 0xF   # high nibble
        vec[i*2+1] = code & 0xF           # low nibble
    return vec

def decode(vec: np.ndarray, original_len: int) -> str:
    """Integer vector -> text"""
    chars = []
    for i in range(original_len):
        high = int(vec[i*2])   & 0xF
        low  = int(vec[i*2+1]) & 0xF
        code = (high << 4) | low
        chars.append(chr(code))
    return ''.join(chars)
