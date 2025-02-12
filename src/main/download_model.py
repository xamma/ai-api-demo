from diffusers import DiffusionPipeline
import torch

"""
Helper function used to download the model locally.
"""
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.save_pretrained("./models/stable-diffusion-xl-base-1.0")