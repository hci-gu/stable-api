from diffusers import DiffusionPipeline, AutoencoderTiny
from compel import Compel, ReturnedEmbeddingsType

import torch

base = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
base.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_device='cuda', torch_dtype=torch.float16)
base.vae = base.vae.cuda()

# base.set_progress_bar_config(disable=True)


prompt = "A portrait of Putin, realistic, photograph, 4k"
prompt2 = "A portrait of Barack obama, realistic, photograph, 4k"

compel = Compel(
    tokenizer=[base.tokenizer, base.tokenizer_2],
    text_encoder=[base.text_encoder, base.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True]
)

prompt_embeds, pooled_prompt_embeds = compel([prompt])
prompt_embeds2, pooled_prompt_embeds2 = compel([prompt2])

# tokens = base.tokenizer("prompt", padding="max_length", max_length=base.tokenizer.model_max_length, truncation=True, return_tensors="pt")
# tokens2 = base.tokenizer_2("prompt", padding="max_length", max_length=base.tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
# embedding = base.text_encoder(tokens.input_ids.to("mps"))[0]
# embedding2 = base.text_encoder_2(tokens2.input_ids.to("mps"))[0]

print(prompt_embeds)
combined_prompt_embeds = prompt_embeds * 0.5 + prompt_embeds2 * 0.5
combined_pooled_prompt_embeds = pooled_prompt_embeds * 0.5 + pooled_prompt_embeds2 * 0.5


image = base(prompt_embeds=combined_prompt_embeds, pooled_prompt_embeds=combined_pooled_prompt_embeds, num_inference_steps=1, guidance_scale=0.0).images[0]

image.save("output.png")