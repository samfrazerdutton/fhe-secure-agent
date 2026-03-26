"""
Run this after installing to verify your GPU works.
"""
import asyncio
from src import SecureAgent

async def dummy_agent(query: str) -> str:
    return f"Response to: {query}"

async def main():
    print("Testing fhe-secure-agent on your GPU...\n")
    agent = SecureAgent(dummy_agent)
    result = await agent.run("hello world")
    print(f"\n✓ Success: {result}")
    print("Your GPU is ready to run secure agents.")

asyncio.run(main())
