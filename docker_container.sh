#!/bin/bash

src=$HOME/codes/signate
docker_img_name=$1
docker_container_name=$2

docker run -itd --gpus all \
--mount type=bind,source=${src},target=/workspace \
--shm-size=6gb \
--name ${docker_container_name} \
${docker_img_name} \
/bin/bash
