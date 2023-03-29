read -p "input data type. train/val: " data_type
if [ "$data_type" = "train" ]; then
    dir_name="train"
elif [ "$data_type" = "val" ]; then
    dir_name="val"
else
    echo "invalid data-type: ${data_type}"
    exit 1
fi

all_annotations_path="./${dir_name}/annotations"
all_annotations=`ls $all_annotations_path`
cnt=1
for annotation_path in $all_annotations; do
    annotation_path=$all_annotations_path/$annotation_path
    python process_annotation.py --path `realpath $annotation_path` --index $cnt
    (( cnt+=1 ))
done
