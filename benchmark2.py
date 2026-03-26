import time
import numpy as np
import sys
sys.path.insert(0, ".")
import seal
from seal import EncryptionParameters, scheme_type, CoeffModulus
from seal import SEALContext, KeyGenerator, Encryptor, Decryptor
from seal import PlainModulus, BatchEncoder

RUNS = 10

print("=" * 60)
print("  Microsoft SEAL CPU — scaling benchmark")
print("=" * 60)

for poly_deg in [1024, 4096, 8192, 16384]:
    try:
        parms = EncryptionParameters(scheme_type.bfv)
        parms.set_poly_modulus_degree(poly_deg)
        parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_deg))
        parms.set_plain_modulus(PlainModulus.Batching(poly_deg, 16))
        context = SEALContext(parms)
        keygen   = KeyGenerator(context)
        pub_key  = keygen.create_public_key()
        sec_key  = keygen.secret_key()
        encryptor = Encryptor(context, pub_key)
        decryptor = Decryptor(context, sec_key)
        encoder   = BatchEncoder(context)

        plain = encoder.encode([7] + [0] * (poly_deg - 1))

        enc_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            ct = encryptor.encrypt(plain)
            enc_times.append((time.perf_counter() - t0) * 1000)

        dec_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            decryptor.decrypt(ct)
            dec_times.append((time.perf_counter() - t0) * 1000)

        print(f"  N={poly_deg:<6} Encrypt: {np.mean(enc_times):.2f}ms   Decrypt: {np.mean(dec_times):.2f}ms")
    except Exception as e:
        print(f"  N={poly_deg:<6} Error: {e}")

print("=" * 60)
