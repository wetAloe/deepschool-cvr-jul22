#!/bin/bash

IMAGE_NAME=dali
CONTAINER_NAME=dali

# run the container
echo "Starting docker container"
docker run -it --rm \
  --ipc=host \
  --network=host \
  --gpus=all \
  -v `pwd`:/workspace/project \
  -v /home/you/folder/with/images/:/workspace/project/data/images \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME"
