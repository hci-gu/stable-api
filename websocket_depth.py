import asyncio
import websockets
import json
from io import BytesIO
from flask_cors import CORS
import base64
from PIL import Image

from sd_depth import *

sd = SD()
test_image = Image.open("depth.jpg")
sd.generate(test_image, "Cat", seed=0)

async def process_prompt(websocket, path):
    async for message in websocket:
        # Parse the JSON message
        request_data = json.loads(message)
        prompt = request_data["prompt"]
        seed = request_data["seed"]
        control_image = request_data["control_image"]
        control_net_scale = request_data["control_net_scale"]
        control_net_from = request_data["control_net_from"]
        control_net_to = request_data["control_net_to"]
        steps = request_data["steps"]
        cfg = request_data["cfg"]

        # control_image is send as string and needs to be converted to image with PIL
        control_image = control_image.split(",")[1]
        control_image = base64.b64decode(control_image)
        control_image = BytesIO(control_image)
        control_image = Image.open(control_image)
        # generate fake blue image
        # image = Image.new("RGB", control_image.size, "blue")
        image = sd.generate(
            control_image,
            prompt,
            steps=steps,
            cfg=cfg,
            size=512,
            control_net_scale=control_net_scale,
            control_net_from=control_net_from,
            control_net_to=control_net_to,
            seed=seed
        )

        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Send image data back to client
        await websocket.send(img_str)

start_server = websockets.serve(process_prompt, "130.241.23.151", 1338)
# start_server = websockets.serve(process_prompt, "localhost", 1338)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()