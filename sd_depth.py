import torch
import os
import xformers
import time
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

class SD:
  def __init__(self):
    self.controlnet = controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small",
        torch_dtype = torch.float16,
        use_safetensors = True,
        variant = "fp16"
    )

    self.base = StableDiffusionXLControlNetPipeline.from_pretrained(
      "stabilityai/sdxl-turbo",
      controlnet=controlnet,
      torch_dtype=torch.float16,
      variant="fp16",
      safety_checker=None,
      requires_safety_checker=False
    ).to("cuda")
    #self.base.vae = AutoencoderTiny.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_device='cuda', torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=False)
    #self.base.vae = self.base.vae.cuda()

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


  def generate(self, image, prompt, steps=1, cfg=0, size=512, control_net_scale=0.75, control_net_from=0, control_net_to=1, seed=None):
    # if seed is not None:
    #   self.generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # prompt_embeds, pooled_prompt_embeds = self.compel([prompt])
    return self.base(
      prompt=prompt,
      negative_prompt="blurry, worst quality, low quality",
      image=image,
      num_inference_steps=steps,
      guidance_scale=cfg,
      width=size,
      height=size,
      controlnet_conditioning_scale=control_net_scale,
      control_guidance_start=control_net_from,
      control_guidance_end=control_net_to,
      output_type="pil",
      generator=torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None,
      return_dict=False
    )[0][0]


  