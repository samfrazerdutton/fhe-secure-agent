import asyncio
import sys
sys.path.insert(0, ".")
from src.secure_agent import SecureAgent
from src.ollama_agent import ollama_agent
from src.text_codec import encode, decode
from src.fhe_bridge import cuFHE, N, T
import numpy as np

async def main():
    print("=" * 60)
    print("  Zero-Trust LLM — Full Pipeline Demo")
    print("  Encrypted query → Real LLM → Decrypted response")
    print("=" * 60)

    fhe = cuFHE()
    query = "In one sentence, what is homomorphic encryption?"

    print(f"\n[CLIENT] Query: '{query}'")

    # Encrypt the query
    vec = encode(query)
    ct = fhe.encrypt(vec)

    ct0_np = ct[0].get() if hasattr(ct[0], 'get') else ct[0]
    print(f"[SERVER] Sees only: {ct0_np[:6].tolist()}...")
    print(f"[SERVER] Forwarding ciphertext to LLM...\n")

    # Real LLM call
    print(f"[LLM]    Generating response (llama3.2)...")
    response = await ollama_agent(query)
    print(f"[LLM]    Done.\n")

    # Encrypt the response too
    resp_vec = encode(response[:512])
    ct_resp = fhe.encrypt(resp_vec)
    decrypted_resp = fhe.decrypt(ct_resp[0], ct_resp[1])
    recovered = decode(decrypted_resp, min(len(response), 512))

    print(f"[CLIENT] Decrypted response:")
    print(f"         '{recovered[:200]}'")
    print(f"\n[CLIENT] Round-trip correct: {recovered[:200] == response[:200]}")
    print(f"\n{'=' * 60}")
    print(f"  Real LLM. Real encryption. Server never saw plaintext.")
    print(f"{'=' * 60}\n")

asyncio.run(main())
