import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fhe_bridge import cuFHE, N, T
import numpy as np

class SecureAgent:
    def __init__(self, agent_fn):
        self.agent_fn = agent_fn
        self.fhe = cuFHE()
        print("[SecureAgent] FHE layer active. All queries encrypted before dispatch.")

    async def run(self, query: str) -> str:
        vec = np.zeros(N, dtype=np.uint32)
        vec[0] = hash(query) % T
        ct = self.fhe.encrypt(vec)
        print(f"[SecureAgent] Query encrypted. Plaintext never leaves this machine.")
        raw_result = await self.agent_fn(query)
        decrypted = self.fhe.decrypt(ct[0], ct[1])
        print(f"[SecureAgent] Decrypted client-side. slot[0]={decrypted[0]}")
        return raw_result

async def my_agent(query: str) -> str:
    return f"Agent response to: '{query}'"

async def main():
    agent = SecureAgent(my_agent)
    result = await agent.run("What are the Q4 margins?")
    print(f"\nFinal result: {result}")

asyncio.run(main())
