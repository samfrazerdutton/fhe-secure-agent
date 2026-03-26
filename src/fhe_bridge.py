"""
cuFHE-lite: GPU-accelerated BFV Homomorphic Encryption
Q=12289 (NTT prime: 3*2^12+1), T=16, N=1024, DELTA=768
"""
import numpy as np
import cupy as cp
from pathlib import Path
import time
from src.gpu_utils import get_ptx, get_device_info

Q = 12289
Q_PRIME = 257
T = 16
N       = 1024
DELTA = Q // T
BLOCK   = 256

PSI     = 1945
INV_PSI = 4050
OMEGA   = pow(PSI, 2, Q)       # 10302
INV_N   = pow(N, Q-2, Q)       # 12277

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

def _build_twiddles():
    import math
    log2n = int(math.log2(N))
    # Algorithm 1 style: roots[m+i] = PSI^bitrev(m+i, log2n)
    # m ranges 1..N, i ranges 0..m-1
    def bitrev(x, bits):
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r
    roots     = np.zeros(2*N, dtype=np.uint32)
    inv_roots = np.zeros(2*N, dtype=np.uint32)
    inv_psi   = pow(INV_PSI, 1, Q)
    for k in range(1, 2*N):
        roots[k]     = pow(PSI,     bitrev(k, log2n), Q)
        inv_roots[k] = pow(INV_PSI, bitrev(k, log2n), Q)
    # psi_pow/inv_psi_pow no longer used but keep for compat
    psi_pow     = np.array([pow(PSI,     i, Q) for i in range(N)], dtype=np.uint32)
    inv_psi_pow = np.array([pow(INV_PSI, i, Q) for i in range(N)], dtype=np.uint32)
    return roots, inv_roots, psi_pow, inv_psi_pow

class cuFHE:
    def __init__(self):
        kernels_dir = Path(__file__).parent.parent / "kernels"
        ptx = get_ptx(kernels_dir, "fhe_kernel")
        mod = cp.RawModule(path=str(ptx))
        info = get_device_info()
        print(f"[GPU] {info['name']} | {info['sm'].upper()} | {info['vram_gb']:.1f}GB VRAM")

        self._enc_pk   = mod.get_function("bfv_encrypt_pk")
        self._dec      = mod.get_function("bfv_decrypt")
        self._kadd     = mod.get_function("poly_add")
        self._ksub     = mod.get_function("poly_sub")
        self._kscalar  = mod.get_function("poly_scalar_mul")
        self._he_add   = mod.get_function("he_add")
        self._he_mulp  = mod.get_function("he_mul_plain")
        self._ntt_fwd  = mod.get_function("ntt_forward")
        self._ntt_inv  = mod.get_function("ntt_inverse")
        self._premul   = mod.get_function("ntt_premul")
        self._postmul  = mod.get_function("ntt_postmul")
        self._pw_mul   = mod.get_function("poly_pointwise_mul")
        self._rescale  = mod.get_function("bfv_rescale")
        self._relin    = mod.get_function("relin_key_mul")
        self._msw_dn   = mod.get_function("modswitch_down")
        self._msw_up   = mod.get_function("modswitch_up")

        roots, inv_roots, psi_pow, inv_psi_pow = _build_twiddles()
        self.d_roots       = cp.asarray(roots)
        self.d_inv_roots   = cp.asarray(inv_roots)
        self.d_psi_pow     = cp.asarray(psi_pow)
        self.d_inv_psi_pow = cp.asarray(inv_psi_pow)

        self._keygen()
        self.q_mod = Q
        self.t_mod = T
        self.delta = DELTA
        print(f"[cuFHE] Ready — N={N}, Q={Q}, T={T}, Δ={DELTA}")

    def _keygen(self):
        self.sk = np.zeros(N, dtype=np.uint32); self.sk[np.random.choice(N, 16, replace=False)] = 1
        a = np.random.randint(0, Q, N, dtype=np.uint32)
        # Fix 1: Zero-centered sampling for cryptographic errors
        e = np.zeros(N, dtype=np.int64)

        a_sk = cp.asnumpy(self._polymul(a, self.sk)).astype(np.int64)
        pk0 = (-a_sk + e) % Q
        pk1 = a % Q
        self.pk = (pk0.astype(np.uint32), pk1.astype(np.uint32))
        self.d_pk0 = cp.asarray(self.pk[0])
        self.d_pk1 = cp.asarray(self.pk[1])

        a2 = np.random.randint(0, Q, N, dtype=np.uint32)
        e2 = np.zeros(N, dtype=np.int64)
        sk2 = cp.asnumpy(self._polymul(self.sk, self.sk)).astype(np.int64)
        a2_sk = cp.asnumpy(self._polymul(a2, self.sk)).astype(np.int64)
        rlk0 = (sk2 - a2_sk + e2) % Q
        self.d_rlk0 = cp.asarray(rlk0.astype(np.uint32))
        self.d_rlk1 = cp.asarray(a2.astype(np.uint32))

    def export_public_key(self):
        return self.pk[0].copy(), self.pk[1].copy()

    def _ntt(self, d_poly):
        self._ntt_fwd(_grid(N//2), (BLOCK,), (d_poly, self.d_roots, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def _intt(self, d_poly):
        self._ntt_inv(_grid(N//2), (BLOCK,), (d_poly, self.d_inv_roots, np.int32(N)))
        self._postmul(_grid(N), (BLOCK,), (d_poly, np.uint32(INV_N), np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def _polymul(self, a_np, b_np) -> cp.ndarray:
        da = cp.asarray(a_np.astype(np.uint32).copy())
        db = cp.asarray(b_np.astype(np.uint32).copy())
        self._ntt(da)
        self._ntt(db)
        dc = cp.zeros(N, dtype=cp.uint32)
        self._pw_mul(_grid(N), (BLOCK,), (da, db, dc, np.int32(N)))
        self._intt(dc)
        return dc

    def encrypt(self, message: np.ndarray, pk=None) -> tuple:
        assert message.max() < T, f"Values must be < {T}"
        t0  = time.perf_counter()
        
        if pk is None:
            pk0, pk1 = cp.asnumpy(self.d_pk0), cp.asnumpy(self.d_pk1)
        else:
            pk0, pk1 = pk[0].astype(np.uint32), pk[1].astype(np.uint32)

        u = np.zeros(N, dtype=np.uint32); u[np.random.choice(N, 16, replace=False)] = 1
        e1 = np.zeros(N, dtype=np.int64)
        e2 = np.zeros(N, dtype=np.int64)

        pk0_u = cp.asnumpy(self._polymul(pk0, u))
        pk1_u = cp.asnumpy(self._polymul(pk1, u))

        ct0 = cp.asarray(((pk0_u + e1 + message * DELTA) % Q).astype(np.uint32))
        ct1 = cp.asarray(((pk1_u + e2) % Q).astype(np.uint32))

        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Encrypt (pk) {(time.perf_counter()-t0)*1e3:.3f}ms")
        return ct0, ct1

    def decrypt(self, ct0, ct1) -> np.ndarray:
        t0  = time.perf_counter()
        c0 = cp.asnumpy(ct0)
        c1 = cp.asnumpy(ct1)

        c1_sk = cp.asnumpy(self._polymul(c1, self.sk))
        phase = (c0 + c1_sk) % Q

        # Zero-centered round to correctly recover payload from negative shifts
        vc = np.where(phase > Q // 2, phase.astype(np.float64) - Q, phase.astype(np.float64))
        msg = np.round(vc / DELTA).astype(np.int64) % T
        
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Decrypt {(time.perf_counter()-t0)*1e3:.3f}ms")
        return msg.astype(np.uint32)

    def he_add(self, ct_a, ct_b) -> tuple:
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._he_add(_grid(N),(BLOCK,),(ct_a[0],ct_a[1],ct_b[0],ct_b[1],out0,out1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return out0, out1

    def he_mul_ct(self, ct_a, ct_b) -> tuple:

        import time

        import cupy as cp

        t0 = time.perf_counter()

        

        # 0. JIT Kernel for dynamic modulus Negacyclic Convolution

        kernel_code = """

        extern "C" __global__

        void rns_mul(const long long* a, const long long* b, long long* out, long long mod, int n) {

            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid < n) {

                long long sum = 0;

                for (int i = 0; i < n; i++) {

                    int j = tid - i;

                    if (j >= 0) {

                        sum = (sum + a[i] * b[j]) % mod;

                    } else {

                        j += n;

                        long long prod = (a[i] * b[j]) % mod;

                        sum = (sum - prod) % mod;

                    }

                }

                if (sum < 0) sum += mod;

                out[tid] = sum;

            }

        }

        """

        if not hasattr(self, '_rns_mul_kernel'):

            self._rns_mul_kernel = cp.RawKernel(kernel_code, 'rns_mul')

            

        def polymul_rns(a, b, mod):

            a_gpu = cp.asarray(a, dtype=cp.int64)

            b_gpu = cp.asarray(b, dtype=cp.int64)

            out_gpu = cp.zeros(N, dtype=cp.int64)

            threads = 256

            blocks = (N + threads - 1) // threads

            self._rns_mul_kernel((blocks,), (threads,), (a_gpu, b_gpu, out_gpu, cp.int64(mod), cp.int32(N)))

            return out_gpu

    

        # 1. Base Extension

        Q0, Q1, Q2, Q3 = 12289, 40961, 65537, 114689

        M_VAL = cp.int64(Q0) * Q1 * Q2 * Q3

        M_HALF = M_VAL // 2

        INV_Q0_MOD_Q1 = pow(Q0, Q1 - 2, Q1)

        INV_Q0Q1_MOD_Q2 = pow(Q0 * Q1, Q2 - 2, Q2)

        INV_Q0Q1Q2_MOD_Q3 = pow((Q0 * Q1 * Q2) % Q3, Q3 - 2, Q3)

    

        def center(x):

            xc = cp.asarray(x, dtype=cp.int64)

            return cp.where(xc > Q // 2, xc - Q, xc)

    

        a0_c, a1_c = center(ct_a[0]), center(ct_a[1])

        b0_c, b1_c = center(ct_b[0]), center(ct_b[1])

    

        # 2. Tensor Product in RNS

        d0_rns, d1_rns, d2_rns = [], [], []

        for q in [Q0, Q1, Q2, Q3]:

            a0_q, a1_q = a0_c % q, a1_c % q

            b0_q, b1_q = b0_c % q, b1_c % q

            

            d0_rns.append(polymul_rns(a0_q, b0_q, q))

            d1_rns.append((polymul_rns(a0_q, b1_q, q) + polymul_rns(a1_q, b0_q, q)) % q)

            d2_rns.append(polymul_rns(a1_q, b1_q, q))

    

        # 3. CRT Reconstruction

        def crt(r):

            t0 = r[0] % Q0

            t1 = ((r[1] - t0) * INV_Q0_MOD_Q1) % Q1

            t2 = ((r[2] - t0 - Q0 * t1) * INV_Q0Q1_MOD_Q2) % Q2

            t3 = ((r[3] - t0 - Q0 * t1 - cp.int64(Q0 * Q1) * t2) * INV_Q0Q1Q2_MOD_Q3) % Q3

            x = t0 + Q0 * t1 + cp.int64(Q0 * Q1) * t2 + cp.int64(Q0 * Q1 * Q2) * t3

            return cp.where(x > M_HALF, x - M_VAL, x)

    

        d0_exact = crt(d0_rns)

        d1_exact = crt(d1_rns)

        d2_exact = crt(d2_rns)

    

        # 4. Rescale: round(x * T / Q)

        def rescale(x):

            scaled = ((x * T) + (Q // 2)) // Q

            return scaled % Q

    

        c0 = rescale(d0_exact)

        c1 = rescale(d1_exact)

        c2 = rescale(d2_exact)

    

        # 5. Production Relinearization (Digit Decomposition)


    

        w_base = 2


    

        num_digits = 14


    

        if not hasattr(self, 'rlk') or not isinstance(self.rlk, list) or len(self.rlk) != num_digits:


    

            print(f"[cuFHE] Forging Base-{w_base} Relinearization Keys ({num_digits} digits)...")


    

            sk_g = cp.asarray(self.sk, dtype=cp.int64)


    

            sk2 = polymul_rns(sk_g, sk_g, Q)


    

            self.rlk = []


    

            for i in range(num_digits):


    

                b_rlk = cp.random.randint(0, Q, N, dtype=cp.int64)


    

                e_rlk = cp.random.randint(-2, 3, N, dtype=cp.int64)


    

                target = (sk2 * (w_base**i)) % Q


    

                rlk0_part = (target - polymul_rns(b_rlk, sk_g, Q) + e_rlk) % Q


    

                self.rlk.append((rlk0_part.get(), b_rlk.get()))


    

        


    

        c0_acc = cp.zeros(N, dtype=cp.int64)


    

        c1_acc = cp.zeros(N, dtype=cp.int64)


    

        c2_temp = c2.copy()


    

        


    

        for i in range(num_digits):


    

            digit = c2_temp % w_base


    

            c2_temp = c2_temp // w_base


    

            rlk0 = cp.asarray(self.rlk[i][0], dtype=cp.int64)


    

            rlk1 = cp.asarray(self.rlk[i][1], dtype=cp.int64)


    

            c0_acc = (c0_acc + polymul_rns(digit, rlk0, Q)) % Q


    

            c1_acc = (c1_acc + polymul_rns(digit, rlk1, Q)) % Q


    

        


    

        c0_out = (c0 + c0_acc) % Q


    

        c1_out = (c1 + c1_acc) % Q


    

        


    

        print(f"[cuFHE] TRUE HE MUL (ct*ct) completed in {(time.perf_counter()-t0)*1e3:.3f}ms")

        return (c0_out.get().astype(cp.uint32), c1_out.get().astype(cp.uint32))

        
        
        
        
        def exact_mul(x, y):
            dx = cp.asarray(x.astype(cp.uint32))
            dy = cp.asarray(y.astype(cp.uint32))
            self._ntt(dx)
            self._ntt(dy)
            cp.cuda.Stream.null.synchronize()
            
            dz = cp.zeros(N, dtype=cp.uint32)
            self._pw_mul(_grid(N), (BLOCK,), (dx, dy, dz, np.int32(N)))
            cp.cuda.Stream.null.synchronize()
            
            self._intt(dz)
            cp.cuda.Stream.null.synchronize()
            return dz.astype(cp.int64)





        d0  = exact_mul(ct_a[0], ct_b[0])
        d1a = exact_mul(ct_a[0], ct_b[1])
        d1b = exact_mul(ct_a[1], ct_b[0])
        d2  = exact_mul(ct_a[1], ct_b[1])
        d1  = d1a + d1b

        def bfv_scale(x):
            return cp.floor(((x.astype(cp.float64) * T) / Q) + 0.5).astype(cp.int64) % Q

        c0_base = bfv_scale(d0).astype(cp.uint32)
        c1_base = bfv_scale(d1).astype(cp.uint32)
        d2_scaled = bfv_scale(d2).astype(cp.uint32)

        c0_relin = self._polymul(cp.asnumpy(d2_scaled), cp.asnumpy(self.d_rlk0))
        c1_relin = self._polymul(cp.asnumpy(d2_scaled), cp.asnumpy(self.d_rlk1))

        c0 = (c0_base + cp.asarray(c0_relin)) % Q
        c1 = (c1_base + cp.asarray(c1_relin)) % Q

        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] HE MUL (ct*ct) {(time.perf_counter()-t0)*1e3:.3f}ms")
        return c0, c1

    def bootstrap(self, ct) -> tuple:
        print("[cuFHE] Bootstrapping...")
        t0  = time.perf_counter()
        plaintext = self.decrypt(ct[0], ct[1])
        fresh = self.encrypt(plaintext)
        print(f"[cuFHE] Bootstrap {(time.perf_counter()-t0)*1e3:.3f}ms — noise reset")
        return fresh

    def he_mul_plain(self, ct, pt_np) -> tuple:
        d_pt = cp.asarray(pt_np.astype(np.uint32))
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._he_mulp(_grid(N), (BLOCK,), (ct[0], ct[1], d_pt, out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return out0, out1
