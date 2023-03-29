#!/bin/bash

# Images
all_images_path=`realpath "./train_images"`
all_counts=`ls $all_images_path | wc -l`
val_counts=$(( all_counts * 2 / 10 ))
train_counts=$(( all_counts - val_counts ))

cnt=0
mkdir -p train/images
mkdir -p val/images
for path in `ls $all_images_path`; do
    path="${all_images_path}/${path}"
    if [ $cnt -lt $train_counts ]; then
        cp $path `realpath "train/images"`/`basename $path`
    else
        cp $path `realpath "val/images"`/`basename $path`
    fi
    cnt=$((cnt+1))
done

# Annotations
all_annotations_path=`realpath "./train_annotations"`
all_counts=`ls $all_annotations_path | wc -l`
val_counts=$(( all_counts * 2 / 10 ))
train_counts=$(( all_counts - val_counts ))

cnt=0
mkdir -p train/annotations
mkdir -p val/annotations
for path in `ls $all_annotations_path`; do
    path="${all_annotations_path}/${path}"
    if [ $cnt -lt $train_counts ]; then
        cp $path `realpath "train/annotations"`/`basename $path`
    else
        cp $path `realpath "val/annotations"`/`basename $path`
    fi
    cnt=$((cnt+1))
done
