import os
import json
from PIL.Image import open as im_open
from glob import glob

# modern_detection_path = "/home/fummicc1/codes/signate/yolov5/runs/detect/exp/labels"
# old_detection_path = "/home/fummicc1/codes/signate/yolov5/runs/detect/exp2/labels"
modern_detection_path = "/home/fummicc1/codes/signate/emsemble/modern-ret"
old_detection_path = "/home/fummicc1/codes/signate/emsemble/old-ret"
is_emsemble = True
is_yolov5 = False

all_label_results_paths = [list(glob(os.path.join(modern_detection_path, "*.txt"))), list(glob(os.path.join(old_detection_path, "*.txt")))]
with open("sample_submit.json", "r") as f:
    data = json.load(f)
for key in data.keys():    
    data[key] = {}
    
modern_id_map = {
    "0": "0_background",
    "1": "1_overall",
    "2": "4_illustration",
    "3": "5_stamp",
    "4": "6_headline",
    "5": "7_caption",
    "6": "8_textline",
    "7": "9_table"
}

old_id_map = {
    "0": "0_background",
    "1": "1_overall",
    "2": "2_handwritten",
    "3": "3_typography",
    "4": "4_illustration",
    "5": "5_stamp",
}

all_labels = {
    "1": "1_overall",
    "2": "2_handwritten",
    "3": "3_typography",
    "4": "4_illustration",
    "5": "5_stamp",
    "6": "6_headline",
    "7": "7_caption",
    "8": "8_textline",
}

for i, label_results_paths in enumerate(all_label_results_paths):
    is_modern = i == 0
    for label_results_path in label_results_paths:
        file_name = label_results_path.split("/")[-1].split(".")[0]
        # backgroundとtableは除く
        image_data = {
            "1_overall": [],
            "2_handwritten": [],
            "3_typography": [],
            "4_illustration": [],
            "5_stamp": [],
            "6_headline": [],
            "7_caption": [],
            "8_textline": [],
        }
        image_file = os.path.join("test_images", f"{file_name}.jpg")
        img = im_open(image_file).convert("RGB")
        img_width, img_height = max(img.size), max(img.size)
        with open(label_results_path, "r") as label_file:
            for line in label_file.readlines():
                if len(line.split()) == 6:
                    id, center_x, center_y, width, height, _ = map(float, line.split())
                else:
                    id, center_x, center_y, width, height = map(float, line.split())
                id = int(id)
                if is_modern and not is_emsemble and is_yolov5:
                    if id == 0 or id == 1:
                        pass
                    else:
                        id = id + 2                    
                center_x *= img_width
                center_y *= img_height
                width *= img_width
                height *= img_height
                if not is_emsemble:
                    if is_yolov5:
                        maps = all_labels
                    else:
                        maps = modern_id_map if is_modern else old_id_map           
                    if str(id) not in maps:
                        continue
                    class_name = maps[str(id)]                
                else:
                    if id == 9 or id == 0:
                        continue
                    class_name = all_labels[str(id)]
                image_data[class_name].append(
                    list(map(int, [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2]))
                )
        deletes = []
        for key in image_data.keys():
            if image_data[key] == []:
                deletes.append(key)
        for key in deletes:
            del image_data[key]
        data[file_name+".jpg"] = image_data    

with open("submit_yolo.json", "w") as f:
    f.write(json.dumps(data))
