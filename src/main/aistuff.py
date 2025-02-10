from diffusers import DiffusionPipeline
import torch
import logging
from io import BytesIO

# Logger settings
logger = logging.getLogger('StableDiffusion_Pipeline')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device {device}.")

def generate_image(prompt: str):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    
    pipe.safety_checker = None
    pipe.to(device)
    
    image = pipe(prompt=prompt).images[0]
    logger.info("Image created")
    
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr

if __name__ == "__main__":
    generate_image("An hedgehog as a star wars stromtrooper.")