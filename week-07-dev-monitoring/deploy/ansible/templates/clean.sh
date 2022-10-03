#!/bin/bash
set -ue

docker container prune -f
docker image prune -f
