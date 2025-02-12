# AI-API Demo
Quick demo for an API that lets you interact with an Diffusion model (stabilityai/stable-diffusion-xl-base-1.0).  

## Prerequisites
You need a CUDA capable GPU with the CUDA and nvidia-container toolkit installed.  

## Build image and run
```
docker build -t aiapi .
docker run -dp 8000:8000 --name=aiapi --gpus all aiapi
```

**Note:** Oviously the images will be HUGE since the model itself already is like ~7GB big. So you might want to decouple this from the image and create a job that pulls the image from another location like an S3 and creates the folder, where the App can then load the model into GPU.  