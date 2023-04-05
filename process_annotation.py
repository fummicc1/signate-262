import argparse
from PIL.Image import open as pil_open
import os
import json

def label_str_to_num(label: str) -> int:
    return int(label[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    path: str = args.path
    index: str = args.index
    with open(path, "r") as f:
        data = json.load(f)
        for label in data["labels"]:
            category = label["category"]
            box = label["box2d"]
            center_x, center_y, width, height = (box["x2"] + box["x1"]) / 2, (box["y2"] + box["y1"]) / 2, box["x2"] - box["x1"], box["y2"] - box["y1"]
            # 画像のサイズを直接取得
            image_name = path.split("/")[-1].split(".")[0]
            image_path = "/".join(path.split("/")[:-2]) + "/images/" + image_name + ".jpg"
            # img = pil_open(image_path).convert("RGB")
            img_width, img_height = (1600, 1600)
            center_x /= img_width
            width /= img_width
            center_y /= img_height
            height /= img_height
            dir = os.path.dirname(path)
            os.makedirs(os.path.join(dir, "yolov8"), exist_ok=True)
            with open(os.path.join(dir, "yolov8", "{:06d}.txt".format(int(index))), "a") as new_f:
                d = [label_str_to_num(category), center_x, center_y, width, height]
                d = map(str, d)
                new_f.write(" ".join(d))
                new_f.write("\n")    
