import argparse
from PIL.Image import open as pil_open
import os
import json

def label_str_to_num(label: str, is_modern: bool, is_all: bool = False) -> int:    
    if is_all:
        m = {
            0: "1_overall",
            1: "2_handwritten",
            2: "3_typography",
            3: "4_illustration",
            4: "5_stamp",
            5: "6_headline",
            6: "7_caption",
            7: "8_textline",
            8: "9_table",
        }
    elif is_modern:
        m = {
            0: "1_overall",
            1: "4_illustration",
            2: "5_stamp",
            3: "6_headline",
            4: "7_caption",
            5: "8_textline",
            6: "9_table",
        }
    else:
        m = {
            0: "1_overall",
            1: "2_handwritten",
            2: "3_typography",
            3: "4_illustration",
            4: "5_stamp",
        }    
    
    for k, v in m.items():
        if v == label:
            return k
    return -1

if __name__ == "__main__":
    IS_ALL = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    path: str = args.path
    index: str = args.index
    with open(path, "r") as f:
        data = json.load(f)
        is_modern = data["attributes"]["年代"] == "近代"
        for label in data["labels"]:
            category = label["category"]
            box = label["box2d"]
            center_x, center_y, width, height = (box["x2"] + box["x1"]) / 2, (box["y2"] + box["y1"]) / 2, box["x2"] - box["x1"], box["y2"] - box["y1"]
            # 画像のサイズを直接取得
            image_name = path.split("/")[-1].split(".")[0]
            image_path = "/".join(path.split("/")[:-2]) + "/images/" + image_name + ".jpg"
            img = pil_open(image_path).convert("RGB")
            img_width, img_height = img.size
            center_x /= img_width
            width /= img_width
            center_y /= img_height
            height /= img_height
            dir = os.path.dirname(path)
            os.makedirs(os.path.join(dir, "..", "labels"), exist_ok=True)
            with open(os.path.join(dir, "..", "labels", "{:06d}.txt".format(int(index))), "a") as new_f:
                label_id = label_str_to_num(category, is_modern, is_all=IS_ALL)
                if label_id == -1:
                    print("invalid category", category)
                d = [label_id, center_x, center_y, width, height]
                d = map(str, d)
                new_f.write(" ".join(d))
                new_f.write("\n")    
