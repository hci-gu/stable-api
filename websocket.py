import asyncio
import websockets
import json
from io import BytesIO
from flask_cors import CORS
import base64

from sd import *

sd = SD()
sd.generate("Cat", seed=0)

async def process_prompt(websocket, path):
    async for message in websocket:
        # Parse the JSON message
        request_data = json.loads(message)
        ps = request_data["prompts"]
        ws = request_data["weights"]
        seed = request_data["seed"]

        # Process the prompt to generate an image
        # This is a placeholder for your image processing logic
        image = sd.generateFromWeightedTextEmbeddings([(p, w) for (p, w) in zip(ps, ws)], neg_prompt="", seed=seed, cfg=0, steps=1, size=512)

        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Send image data back to client
        await websocket.send(img_str)

start_server = websockets.serve(process_prompt, "130.241.23.151", 1338)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()