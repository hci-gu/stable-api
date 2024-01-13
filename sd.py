import torch
import os
import xformers
import time
from diffusers import DiffusionPipeline, AutoencoderTiny
from compel import Compel, ReturnedEmbeddingsType
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

class SD:
  def __init__(self):
    self.base = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", safety_checker=None, requires_safety_checker=False).to("cuda")
    self.base.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
    self.base.vae = self.base.vae.cuda()

    config = CompilationConfig.Default()
    config.enable_xformers = True
    # config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = False
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True

    self.base = compile(self.base, config)
    self.base.set_progress_bar_config(disable=True)
    self.compel = Compel(
      tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
      text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
      returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
      requires_pooled=[False, True]
    )

  def weightedEmbeds(self, prompt, weight):
    prompt_embeds, pooled_prompt_embeds = self.compel([prompt])
    print("compel")
    print(prompt_embeds)
    print(pooled_prompt_embeds)
    print(pooled_prompt_embeds.shape)

    return (prompt_embeds * weight, pooled_prompt_embeds * weight)
  
  def customEmbedding(self, prompt, weight):
    # tokens = self.base.tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest").to("cuda")
    # prompt_embeds = self.base.text_encoder(
    #     **tokens, output_hidden_states=True
    # )
    # # create pooled prompt embeds
    # pooled_prompt_embeds = prompt_embeds.hidden_states[-1].mean(1)
    # # multiply by weight

    # # pooled_prompt_embeds = prompt_embeds[0]

    # prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    response = self.base.encode_prompt(prompt)
    print(response)
    prompt_embeds = response[0]
    pooled_prompt_embeds = response[2]

    print("custom")
    print(prompt_embeds)
    print(pooled_prompt_embeds)
    print(pooled_prompt_embeds.shape)

    # bs_embed, seq_len, _ = prompt_embeds.shape
    # prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    # prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    return (prompt_embeds * weight, pooled_prompt_embeds * weight)

    # tokens = self.base.tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest").to("cuda")
    # encoded = self.base.text_encoder(**tokens)
    # print(encoded)
    
    # return (hidden_states * weight, pooled_hidden_states * weight)
  
  # inputs is list of tuples (prompt, weight)
  def generateFromWeightedTextEmbeddings(self, inputs, neg_prompt="", steps=1, cfg=0, size=512, seed=None):

    text_embeddings = [self.customEmbedding(p, w) for (p, w) in inputs]
    prompt_embeddings = torch.stack([t[0] for t in text_embeddings]).sum(0)
    pooled_prompt_embeddings = torch.stack([t[1] for t in text_embeddings]).sum(0)

    return self.base(prompt_embeds=prompt_embeddings, pooled_prompt_embeds=pooled_prompt_embeddings, num_inference_steps=steps, guidance_scale=cfg, width=size, height=size, output_type="pil", generator=torch.Generator(device="cuda").manual_seed(seed), return_dict=False)[0][0]

  def generateFromWeightedTextEmbeddingsDebug(self, inputs, neg_prompt="", steps=1, cfg=0, size=512, seed=None):
    # if seed is not None:
    #   self.generator = torch.Generator(device="cuda").manual_seed(seed)

    # text_embeddings = [self.weightedEmbeds(p, w) for (p, w) in inputs]
    start = time.time()
    text_embeddings = [self.customEmbedding(p, w) for (p, w) in inputs]
    text_embeddings_time = time.time() - start
    start = time.time()
    prompt_embeddings = torch.stack([t[0] for t in text_embeddings]).sum(0)
    prompt_embeddings_time = time.time() - start
    pooled_prompt_embeddings = torch.stack([t[1] for t in text_embeddings]).sum(0)
    start = time.time()
    pooled_prompt_embeddings_time = time.time() - start
    start = time.time()
    image = self.base(prompt_embeds=prompt_embeddings, pooled_prompt_embeds=pooled_prompt_embeddings, num_inference_steps=steps, guidance_scale=cfg, width=size, height=size, output_type="pil", generator=torch.Generator(device="cuda").manual_seed(seed), return_dict=False)[0][0]
    image_time = time.time() - start

    times = {
      'text_embeddings': text_embeddings_time,
      'prompt_embeddings': prompt_embeddings_time,
      'pooled_prompt_embeddings': pooled_prompt_embeddings_time,
      'image': image_time
    }

    return image, times



  def generate(self, prompt, steps=1, cfg=0, size=512, seed=None):
    # if seed is not None:
    #   self.generator = torch.Generator(device="cuda").manual_seed(seed)
    
    prompt_embeds, pooled_prompt_embeds = self.compel([prompt])
    return self.base(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, num_inference_steps=steps, guidance_scale=cfg, width=size, height=size, output_type="pil",generator=torch.Generator(device="cuda").manual_seed(seed), return_dict=False)[0][0]


  # def generate(self, prompt, neg_prompt="", steps=20, cfg=7.5, seed=None):
  #   if seed is not None:
  #     self.generator.manual_seed(seed)
  #   c_embedding = self.getTextEmbedding(prompt)
  #   u_embedding = self.getTextEmbedding(neg_prompt)
  #   text_embeddings = torch.cat([u_embedding, c_embedding])
  #   latents = self.generateLatents()
  #   latents = self.runSteps(latents, text_embeddings, steps=steps, cfg=cfg)
  #   latents = 1 / 0.18215 * latents
  #   image = self.vae.decode(latents).sample
  #   # create PIL image
  #   image = (image / 2 + 0.5).clamp(0, 1)
  #   image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  #   images = (image * 255).round().astype("uint8")
  #   pil_images = [Image.fromarray(image) for image in images]
  #   return pil_images[0]
