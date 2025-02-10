FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip

COPY src/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

COPY src/torch_requirements.txt /opt/torch_requirements.txt
RUN pip install --no-cache-dir -r /opt/torch_requirements.txt

COPY src/main /app

EXPOSE 8000

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["fastapi", "run", "--port", "8000"]
