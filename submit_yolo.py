import os
import json
from PIL.Image import open as im_open
from glob import glob

mordern_detection_path = "runs/detect/predict3/labels"
old_detection_path = "runs/detect/predict4/labels"

label_results_paths = list(glob(os.path.join(mordern_detection_path, "*.txt"))) + list(glob(os.path.join(old_detection_path, "*.txt")))
data = {}

for label_results_path in label_results_paths:
    print(label_results_path)
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
    img_width, img_height = img.size
    with open(label_results_path, "r") as label_file:
        for line in label_file.readlines():
            id, center_x, center_y, width, height = map(float, line.split())
            id = int(id)
            center_x *= img_width
            center_y *= img_height
            width *= img_width
            height *= img_height
            class_name = ""
            for key in image_data.keys():
                if key.startswith(str(id)):
                    class_name = key
                    break
            if class_name == "":
                continue
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
