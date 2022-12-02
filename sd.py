import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from PIL import Image
import os

huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

class SD:
  def __init__(self):
#    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
#    pipe = pipe.to("cuda")
    self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=huggingface_token)
    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=huggingface_token)
    self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    self.device = "cuda"
    self.vae.to(self.device)
    self.text_encoder.to(self.device)
    self.unet.to(self.device)
    self.generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

  def generateLatents(self, height=512, width=512):
    latents = torch.randn(
        (1, self.unet.in_channels, height // 8, width // 8),
        generator=self.generator,
    )
    latents = latents.to(self.device)
    return latents * self.scheduler.init_noise_sigma

  def runSteps(self, latents, embedding, steps=20, cfg=7.5):
    self.scheduler.set_timesteps(steps)

    for t in tqdm(self.scheduler.timesteps):
        print(f'step {t}')
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embedding).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    return latents

  def getTextEmbedding(self, prompt):
    text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
    return text_embeddings

  def getUnconditionedEmbedding(self, max_length):
    uncond_input = self.tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]  
    return uncond_embeddings

  # inputs is list of tuples (prompt, weight)
  def generateFromWeightedTextEmbeddings(self, inputs, neg_prompt="", steps=20, cfg=7.5, seed=None):
    if seed is not None:
      self.generator.manual_seed(seed)
    c_embedding = torch.stack([self.getTextEmbedding(prompt) * weight for (prompt, weight) in inputs]).sum(0)
    u_embedding = self.getTextEmbedding(neg_prompt)
    text_embeddings = torch.cat([u_embedding, c_embedding])
    latents = self.generateLatents()
    latents = self.runSteps(latents, text_embeddings, steps=steps, cfg=cfg)
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    # create PIL image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]

  def generate(self, prompt, neg_prompt="", steps=20, cfg=7.5, seed=None):
    if seed is not None:
      self.generator.manual_seed(seed)
    c_embedding = self.getTextEmbedding(prompt)
    u_embedding = self.getTextEmbedding(neg_prompt)
    text_embeddings = torch.cat([u_embedding, c_embedding])
    latents = self.generateLatents()
    latents = self.runSteps(latents, text_embeddings, steps=steps, cfg=cfg)
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    # create PIL image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]
