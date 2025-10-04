import aiohttp
import asyncio

async def send_chunks_mcp(chunks, source_description="Uploaded file"):
    url = "http://localhost:11434/api/episodes"
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk in enumerate(chunks):
            payload = {
                "name": f"Chunk {i}",
                "episode_body": chunk,
                "source": "text",
                "source_description": source_description,
            }
            tasks.append(session.post(url, json=payload))
        responses = await asyncio.gather(*tasks)
        for r in responses:
            if r.status != 200:
                print(f"[ERROR] Failed to upload chunk: {await r.text()}")
            else:
                print(f"[INFO] Chunk uploaded successfully")

# Usage
# asyncio.run(send_chunks_mcp(chunks))
