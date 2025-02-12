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

pipe = None

def load_model():
    """Load and return the model."""
    global pipe
    if pipe is None:
        model_path = "models/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        )
        pipe.safety_checker = None
        pipe.to(device)
        logger.info("Model loaded successfully from local storage.")
    return pipe

def generate_image(prompt: str):
    """Generate an image using the preloaded model."""
    if pipe is None:
        raise ValueError("Model is not loaded yet. Call `load_model()` first.")
    
    logger.info(f"Generating image for prompt: {prompt}")
    image = pipe(prompt=prompt).images[0]
    
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    
    logger.info("Image created successfully.")
    return img_byte_arr
