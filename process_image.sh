#!/bin/bash

mode="train"
dir=`dirname $0`/$mode/images
cnt=1
for filename in `ls $dir`; do
    path=$dir/$filename
    echo "path: ${path}"
    format_cnt=$(printf "%06d" $cnt)
    mv $path $dir/$format_cnt.jpg
    (( cnt += 1 ))
done
