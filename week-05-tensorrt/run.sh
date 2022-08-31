#!/bin/bash

IMAGE_NAME=trt
CONTAINER_NAME=trt

# run the container
echo "Starting docker container"
docker run -it --rm \
  --ipc=host \
  --network=host \
  --gpus=all \
  -v `pwd`:/workspace/project \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME"
