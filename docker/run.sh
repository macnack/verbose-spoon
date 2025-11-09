#!/bin/bash

docker run -it --rm \
    --hostname="$(hostname)" \
    --ipc=host \
    --gpus all \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --volume="$(pwd)/..:/home/mackop/workspace" \
    --privileged \
    --network=host \
    --name=gps_denied \
    --user "$(id -u):$(id -g)" \
    edm "/bin/bash"
