#!/bin/bash

path=$HOME/codes/signate
tag_name=$1

docker build $path --tag $tag_name
