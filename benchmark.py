"""
Benchmark: fhe-secure-agent (GPU) vs Microsoft SEAL (CPU)
"""
import time
import numpy as np
import sys
sys.path.insert(0, ".")
from src.fhe_bridge import cuFHE, N, T

print("=" * 60)
print("  fhe-secure-agent vs Microsoft SEAL — Benchmark")
print("=" * 60)

RUNS = 20

# ── GPU (your implementation) ──────────────────────────────
print("\n[1/2] Warming up GPU...")
fhe = cuFHE()
vec = np.zeros(N, dtype=np.uint32)
vec[0] = 7

# warmup
ct = fhe.encrypt(vec)
fhe.decrypt(ct[0], ct[1])

print(f"[1/2] Benchmarking GPU ({RUNS} runs)...")
enc_times, dec_times = [], []
for _ in range(RUNS):
    t0 = time.perf_counter()
    ct = fhe.encrypt(vec)
    enc_times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    fhe.decrypt(ct[0], ct[1])
    dec_times.append((time.perf_counter() - t0) * 1000)

gpu_enc = np.mean(enc_times)
gpu_dec = np.mean(dec_times)

# ── CPU (Microsoft SEAL) ───────────────────────────────────
print(f"\n[2/2] Benchmarking Microsoft SEAL CPU ({RUNS} runs)...")
import seal
from seal import EncryptionParameters, scheme_type, CoeffModulus, PlainModulus
from seal import SEALContext, KeyGenerator, Encryptor, Decryptor, BatchEncoder

parms = EncryptionParameters(scheme_type.bfv)
parms.set_poly_modulus_degree(1024)
parms.set_coeff_modulus(CoeffModulus.BFVDefault(1024))
parms.set_plain_modulus(PlainModulus.Batching(1024, 16))
context = SEALContext(parms)

keygen   = KeyGenerator(context)
pub_key  = keygen.create_public_key()
sec_key  = keygen.secret_key()
encryptor = Encryptor(context, pub_key)
decryptor = Decryptor(context, sec_key)
encoder   = BatchEncoder(context)

plain_vec = [7] + [0] * (1023)
plain = encoder.encode(plain_vec)

seal_enc_times, seal_dec_times = [], []
for _ in range(RUNS):
    t0 = time.perf_counter()
    ct_seal = encryptor.encrypt(plain)
    seal_enc_times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    decryptor.decrypt(ct_seal)
    seal_dec_times.append((time.perf_counter() - t0) * 1000)

seal_enc = np.mean(seal_enc_times)
seal_dec = np.mean(seal_dec_times)

# ── Results ────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  Results (average of {RUNS} runs, N=1024)")
print(f"{'=' * 60}")
print(f"  {'Operation':<12} {'fhe-secure-agent (GPU)':>22} {'SEAL (CPU)':>12} {'Speedup':>10}")
print(f"  {'-'*58}")
print(f"  {'Encrypt':<12} {gpu_enc:>19.2f}ms {seal_enc:>10.2f}ms {seal_enc/gpu_enc:>9.1f}x")
print(f"  {'Decrypt':<12} {gpu_dec:>19.2f}ms {seal_dec:>10.2f}ms {seal_dec/gpu_dec:>9.1f}x")
print(f"{'=' * 60}")
print(f"\n  GPU: RTX 2060 (SM_75) vs CPU: {RUNS} run average")
print(f"  BFV N=1024, Q=12289, T=16\n")
