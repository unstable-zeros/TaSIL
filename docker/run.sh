#!/usr/bin/env bash
PROJECT=sil

EXT_UID=$(id -u)
EXT_GID=$(id -g)
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p .docker_home
docker run --rm \
    --gpus ${1-all} \
    --env EXT_USER=$USER --env EXT_UID=$EXT_UID --env EXT_GID=$EXT_GID \
    -v $DIR/../.docker_home:/home/user \
    -v $DIR/..:/home/user/code \
    -h docker \
    -it $USER-$PROJECT-container:latest
