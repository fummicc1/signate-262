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

dir=$(realpath `dirname $0`)
ann=$dir/$mode/annotations
annotations=`ls $ann`
cnt=1
for annotation in $annotations; do
    num=$(printf "%06d" $cnt)
    mv $ann/$annotation $ann/$num.json
    ((cnt++))
done
