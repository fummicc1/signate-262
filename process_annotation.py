import argparse
import os
import json

def label_str_to_num(label: str) -> int:
    return int(label[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    path = args.path
    index = args.index
    with open(path, "r") as f:
        data = json.load(f)
        for label in data["labels"]:
            category = label["category"]
            box = label["box2d"]
            center_x, center_y, width, height = (box["x2"] + box["x1"]) / 2, (box["y2"] + box["y1"]) / 2, box["x2"] - box["x1"], box["y2"] - box["y1"]
            dir = os.path.dirname(path)
            os.makedirs(os.path.join(dir, "yolov8"), exist_ok=True)
            with open(os.path.join(dir, "yolov8", "{:06d}".format(int(index))), "a") as new_f:
                d = [label_str_to_num(category), center_x, center_y, width, height]
                d = map(str, d)
                new_f.write(" ".join(d))
                new_f.write("\n")
