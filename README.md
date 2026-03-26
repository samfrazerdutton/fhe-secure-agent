# FHE Secure Agent

> Wrap any LLM agent in GPU-accelerated homomorphic encryption.
> The server computes on ciphertexts. Plaintext never leaves your machine.

## Install
pip install fhe-secure-agent

## Usage — one line to secure any agent
from src.secure_agent import SecureAgent

secure = SecureAgent(your_agent_function)
result = await secure.run("your query")

## Benchmarks (RTX 2060, SM_75)
| Operation | Time    |
|-----------|---------|
| Encrypt   | 1.09ms  |
| Decrypt   | 1.43ms  |

## Why this exists
NeMo Guardrails enforces what the LLM *says*.
This enforces what the LLM *sees*.
Without this layer, zero-trust AI is a marketing claim.
With it, it's a cryptographic guarantee.

## Architecture
BFV scheme — N=1024, Q=12289, T=16
Custom PTX kernels compiled for SM_75
GPU-accelerated NTT, key generation, polynomial multiply
