import asyncio
import json
import websockets


async def main():
    uri = "ws://localhost:8000/ws/query"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"query": "What is the company policy on pet adoption leave?"}))
        while True:
            message = await ws.recv()
            event = json.loads(message)
            if event["type"] == "trace":
                data = event["data"]
                summary = {k: v for k, v in data.items() if k != "timestamp"}
                print(f"  [{event['node']}] {summary}")
            elif event["type"] == "final":
                print("\nFINAL:")
                print(" answer:", event["data"]["answer"][:200])
                print(" used_web_fallback:", event["data"]["used_web_fallback"])
                break
            elif event["type"] == "error":
                print("ERROR:", event["data"])
                break


asyncio.run(main())
