#!/bin/bash

docker run -it --rm \
    --env-file .secret \
    --hostname="$(hostname)" \
    --ipc=host \
    --gpus all \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --volume="$(pwd)/..:/home/mackop/workspace" \
    --privileged \
    --network=host \
    --name=gps_denied_exp \
    --user "$(id -u):$(id -g)" \
    --volume="/media/mackop/data_ssd://home/mackop/workspace/data/poznan" \
    edm:exp "/bin/bash"
