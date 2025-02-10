# AI-API Demo
Quick demo for an API that lets you interact with an Diffusion model (stabilityai/stable-diffusion-xl-base-1.0).  

## Prerequisites
You need a CUDA capable GPU with the CUDA and nvidia-container toolkit installed.  

## Build image and run
```
docker build -t aiapi .
docker run -dp 8000:8000 --name=aiapi --gpus all aiapi
```