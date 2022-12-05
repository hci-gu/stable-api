from flask import Flask, send_file, request
import torch
from flask_cors import CORS

from sd import *

sd = SD()
sd2 = SD(v=2)
app = Flask(__name__)
CORS(app)

@app.route('/img/<prompt>')
def img(prompt):
  args = request.args

  steps = int(args.get('steps', default=10))
  cfg = float(args.get('cfg', default=7.5))
  seed = int(args.get('seed'))
  neg_prompt = args.get('neg_prompt', default="")

  if args.get('v2', default=None) is None:
    image = sd.generate(prompt, steps=steps, cfg=cfg, seed=seed, neg_prompt=neg_prompt)
  else:
    image = sd2.generate(prompt, steps=steps, cfg=cfg, seed=seed, neg_prompt=neg_prompt)

  image.save("output.png")
  return send_file('./output.png')

@app.route('/combine', methods=['POST'])
def combine():
  args = request.args
  ps = request.json['prompts']
  ws = request.json['weights']
  cfg = request.json['cfg'] if 'cfg' in request.json else 7.5
  steps = request.json['steps'] if 'steps' in request.json else 20
  seed = request.json['seed'] if 'seed' in request.json else None
  neg_prompt = request.json['neg_prompt'] if 'neg_prompt' in request.json else ""

  if 'v2' in request.json:
    image = sd2.generateFromWeightedTextEmbeddings([(p, w) for (p, w) in zip(ps, ws)], neg_prompt=neg_prompt, seed=seed, cfg=cfg, steps=steps)
  else:
    image = sd.generateFromWeightedTextEmbeddings([(p, w) for (p, w) in zip(ps, ws)], neg_prompt=neg_prompt, seed=seed, cfg=cfg, steps=steps)
    

  image.save("output.png")
  return send_file('./output.png')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
