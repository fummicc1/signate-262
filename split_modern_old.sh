#!/bin/bash

set -e
set +x

cd $(dirname $0)

read -p "select mode. (train/val) : " mode
if [ "$mode" = "train" ]; then
    echo "train mode"
elif [ "$mode" = "val" ]; then
    echo "val mode"
else
    echo "invalid mode: ${mode}"
    exit 1
fi
annotations=$(ls train_annotations)
target_path=$(ls ${mode}/images)
cnt=0
if [ $mode = "val" ]; then
    train_cnt=`ls train/images | wc -w`
    cnt=$(( cnt += $train_cnt ))
fi
for target in $target_path; do    
    cnt=$((cnt+1))
    target=`realpath "$mode/images/$target"`
    label=`realpath "$mode/labels/$(basename $target .jpg).txt"`
    annotation=$(realpath train_annotations/$(echo "${annotations}" | sed -n ${cnt}p))    
    old_token="\"年代\": \"古典籍\""
    modern_token="\"年代\": \"近代\""
    old_token_cnt=`cat "$annotation" | grep "$old_token" | wc -l`
    modern_token_cnt=`cat "$annotation" | grep "$modern_token" | wc -l`
    filename=`basename $target .jpg`
    if [ $old_token_cnt -gt 0 ]; then
        img_dir="$mode/old/images"
        label_dir="$mode/old/labels"
        annotation_dir="$mode/old/annotations"
    elif [ $modern_token_cnt -gt 0 ]; then
        img_dir="$mode/modern/images"
        label_dir="$mode/modern/labels"
        annotation_dir="$mode/modern/annotations"
    else
        echo skipped
        continue
    fi
    [ -d $img_dir ] || mkdir -p $img_dir
    [ -d $label_dir ] || mkdir -p $label_dir
    [ -d $annotation_dir ] || mkdir -p $annotation_dir    
    cp $target $img_dir/$filename.jpg
    cp $label $label_dir/$filename.txt
    cp $annotation $annotation_dir/$filename.json
done
