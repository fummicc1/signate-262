#!/bin/bash

read -p "mode. train/val: " mode
if [ "$mode" = "train" ]; then
    dir_name="train"
elif [ "$mode" = "val" ]; then
    dir_name="val"
else
    echo "invalid data-type: ${mode}"
    exit 1
fi

dir=`dirname $0`/$mode/images
cnt=1
for filename in `ls $dir`; do
    path=$dir/$filename
    echo "path: ${path}"
    format_cnt=$(printf "%06d" $cnt)
    mv $path $dir/$format_cnt.jpg
    (( cnt += 1 ))
done
