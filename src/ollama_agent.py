import httpx

async def ollama_agent(query: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": query,
                "stream": False
            }
        )
        data = response.json()
        return data["response"]
