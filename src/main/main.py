from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import logging
import threading
import asyncio
from queue import Queue
from aistuff import generate_image, load_model

app = FastAPI()

"""Logger settings"""
logger = logging.getLogger('API_Logger')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

"""Global model reference and task queue"""
model_ready = threading.Event()
task_queue = Queue()

def initialize_model():
    """Load model once in the background."""
    global model_ready
    if not model_ready.is_set():
        load_model()
        model_ready.set()
        logger.info("Model is ready for inference!")

"""Start model loading in a background thread"""
threading.Thread(target=initialize_model, daemon=True).start()

"""
Async Task to handle image generation concurrently.
Needed since the generate_image is not async.
"""
async def generate_image_task(prompt: str):
    """Generate image in a background thread."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, generate_image, prompt)
    return result

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate-image")
async def generate_image_endpoint(prompt: str):
    # check if the model is ready
    if not model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still loading, please wait.")

    # only allow processing of one requests, since model isnt thread safe
    if not task_queue.empty():
        raise HTTPException(status_code=429, detail="Another image is being processed. Please wait for it to finish.")

    logger.info(f"Received request to generate image for prompt: {prompt}")

    # add the request to the task queue = mark as in progress
    task_queue.put(prompt)

    try:
        # generate image with task fnction
        image_bytes = await generate_image_task(prompt)
    finally:
        # task finished, remove from queue
        task_queue.get()

    # Byte stream response
    return StreamingResponse(image_bytes, media_type="image/png")
