from flask import Flask, send_file, request, jsonify
import asyncio
import websockets
import json
from io import BytesIO
from flask_cors import CORS
import time
import base64

from sd import *

sd = SD()
app = Flask(__name__)
CORS(app)

@app.route('/combine', methods=['POST'])
def combine():
  ps = request.json['prompts']
  ws = request.json['weights']
  cfg = request.json['cfg'] if 'cfg' in request.json else 0
  steps = request.json['steps'] if 'steps' in request.json else 1
  seed = request.json['seed'] if 'seed' in request.json else None
  neg_prompt = request.json['neg_prompt'] if 'neg_prompt' in request.json else ""
  size = request.json['size'] if 'size' in request.json else 512

  image = sd.generateFromWeightedTextEmbeddings([(p, w) for (p, w) in zip(ps, ws)], neg_prompt=neg_prompt, seed=seed, cfg=cfg, steps=steps, size=size)

  img_io = BytesIO()
  image.save(img_io, 'JPEG', quality=70)
  img_io.seek(0)
  return send_file(img_io, mimetype='image/jpeg')

@app.route('/generate', methods=['POST'])
def generate():
  prompt = request.json['prompt']
  cfg = request.json['cfg'] if 'cfg' in request.json else 0
  steps = request.json['steps'] if 'steps' in request.json else 1
  seed = request.json['seed'] if 'seed' in request.json else None
  size = request.json['size'] if 'size' in request.json else 512

  image = sd.generate(prompt, seed=seed, cfg=cfg, steps=steps, size=size)

  img_io = BytesIO()
  image.save(img_io, 'JPEG', quality=70)
  img_io.seek(0)
  return send_file(img_io, mimetype='image/jpeg')

# same as combine but debug how long everything takes
@app.route('/combine-debug', methods=['POST'])
def combineDebug():
  # timestamp when we start
  request_start = time.time()
  ps = request.json['prompts']
  ws = request.json['weights']
  cfg = request.json['cfg'] if 'cfg' in request.json else 0
  steps = request.json['steps'] if 'steps' in request.json else 1
  seed = request.json['seed'] if 'seed' in request.json else None
  neg_prompt = request.json['neg_prompt'] if 'neg_prompt' in request.json else ""
  size = request.json['size'] if 'size' in request.json else 512

  (image, times) = sd.generateFromWeightedTextEmbeddingsDebug([(p, w) for (p, w) in zip(ps, ws)], neg_prompt=neg_prompt, seed=seed, cfg=cfg, steps=steps, size=size)
  start = time.time()
  img_io = BytesIO()
  image.save(img_io, 'JPEG', quality=70)
  img_byte_arr = img_io.getvalue()
  encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
  image_save = time.time() - start
  # send back image and all timestamps
  return jsonify({
    'image': encoded_image,
    'request_start': request_start,
    'text_embeddings': times['text_embeddings'],
    'prompt_embeddings': times['prompt_embeddings'],
    'pooled_prompt_embeddings': times['pooled_prompt_embeddings'],
    'generate': times['image'],
    'image_save': image_save
  })

  

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

app.run(debug=True, host='0.0.0.0', threaded=False, port=4000)
