import asyncio
import websockets
import json

async def test():
    uri = 'ws://localhost:8000/api/ae/training/ws/ae-job-8f2a-439b'
    try:
        async with websockets.connect(uri) as websocket:
            print('Connected to', uri)
            await websocket.send('START')
            print('Sent START')
            for _ in range(5):
                response = await websocket.recv()
                print('Received:', response)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
