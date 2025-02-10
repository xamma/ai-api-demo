from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import logging
from aistuff import generate_image

app = FastAPI()

# Logger settings
logger = logging.getLogger('API_LOgger')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post("/generate-image/")
async def generate_image_endpoint(prompt: str):
    image_bytes = generate_image(prompt)
    return StreamingResponse(image_bytes, media_type="image/png")