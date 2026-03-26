# fhe-secure-agent

> Wrap any LLM agent in GPU-accelerated homomorphic encryption.  
> The server computes on ciphertexts. Plaintext never leaves your machine.

## The problem with every LLM deployment today

When you send a query to any AI agent, the server sees your plaintext.  
Your financial data. Your medical records. Your IP.  
NeMo Guardrails can enforce what the LLM *says*.  
It cannot enforce what the LLM *sees*.

**This library fixes that.**

## What the server actually sees
```
[CLIENT] Query: 'What are the Q4 financial margins?'
[SERVER] Receives ciphertext — sees only:
         ct[0][:8] = [6930, 9394, 3877, 7819, 10960, 11452, 6804, 5336]
[CLIENT] Round-trip correct: True
Encrypt: 1.1ms  |  Decrypt: 1.4ms  |  GPU: RTX 2060
```

## Install
```bash
pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11
pip install fhe-secure-agent
```

## Usage — one line to secure any agent
```python
from fhe_secure_agent import SecureAgent

async def my_agent(query: str) -> str:
    # your existing agent code here
    return response

secure = SecureAgent(my_agent)
result = await secure.run("What are the Q4 margins?")
```

## Supported GPUs

| Architecture | GPUs |
|---|---|
| SM_60/61 | GTX 10 series |
| SM_70 | Titan V, Tesla V100 |
| SM_75 | RTX 20 series |
| SM_80 | A100 |
| SM_86 | RTX 30 series |
| SM_89 | RTX 40 series |
| SM_90 | H100 |

Auto-detects your GPU. No configuration needed.

## Benchmarks (RTX 2060, SM_75)

| Operation | Time |
|---|---|
| Encrypt | 1.1ms |
| Decrypt | 1.4ms |

## Architecture

- BFV homomorphic encryption scheme
- N=1024, Q=12289, T=16
- Custom PTX kernels compiled per SM architecture
- GPU-accelerated NTT, key generation, polynomial multiply
- RNS basis across 4 primes for ciphertext multiplication
- Digit decomposition relinearization

## Run the demo
```bash
git clone https://github.com/samfrazerdutton/fhe-secure-agent
cd fhe-secure-agent
python demo.py
```

## Author

Sam Frazer-Dutton  
GPU-accelerated cryptography for LLM inference pipelines.
