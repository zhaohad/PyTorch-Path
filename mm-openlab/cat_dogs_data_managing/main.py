import os
import random
import re


def arrange_train():
    path = "./data/train/train"
    cats_path = "./data/train/class_cats"
    dogs_path = "./data/train/class_dogs"
    os.makedirs(cats_path, exist_ok=True)
    os.makedirs(dogs_path, exist_ok=True)
    for root, dirs, files in os.walk(path):
        cnt_cats = 0
        cnt_dogs = 0
        cnt_cls = 0
        for f in files:
            f_path = os.path.join(root, f)
            cls = ""
            cls_root = ""
            if f.startswith("cat"):
                cls = "cats"
                cls_root = cats_path
                cnt_cats += 1
                cnt_cls = cnt_cats
            elif f.startswith("dog"):
                cls = "dogs"
                cls_root = dogs_path
                cnt_dogs += 1
                cnt_cls = cnt_dogs

            new_name = f"{cls}{cnt_cls}.jpg"
            new_path = os.path.join(cls_root, new_name)
            os.rename(f_path, new_path)
            print(f"f_path = {f_path}, new_path = {new_path}")

        print(root)


def construct_val():
    val_path = "./data/val"
    val_cats_path = "./data/val/class_cats"
    val_dogs_path = "./data/val/class_dogs"
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(val_cats_path, exist_ok=True)
    os.makedirs(val_dogs_path, exist_ok=True)
    cnt_val = 100

    cats_path = "./data/train/class_cats"
    dogs_path = "./data/train/class_dogs"

    paths = [val_cats_path, val_dogs_path]
    for cls_path in paths:
        cls = ""

        if cls_path.endswith("cats"):
            data_path = cats_path
            cls = "cats"
        elif cls_path.endswith("dogs"):
            data_path = dogs_path
            cls = "dogs"

        for root, _, files in os.walk(data_path):
            sub_files = random.sample(files, cnt_val)
            cnt_cls = 0
            for f in sub_files:
                cnt_cls += 1
                file_path = os.path.join(root, f)
                new_path = os.path.join(cls_path, f"{cls}{cnt_cls}.jpg")
                os.rename(file_path, new_path)
                print(f"file_path = {file_path}, new_path = {new_path}")


def re_arrange_train():
    train_cats_path = "./data/train/class_cats"
    train_dogs_path = "./data/train/class_dogs"
    clses = ["cats", "dogs"]

    for cls in clses:
        data_path = ""
        if cls.endswith("cats"):
            data_path = train_cats_path
        elif cls.endswith("dogs"):
            data_path = train_dogs_path

        cnt_cls = 0
        for root, _, files in os.walk(data_path):
            for f in files:
                cnt_cls += 1
                f_path = os.path.join(root, f)
                tmp_path = os.path.join(root, f"{cnt_cls}")
                print(f"f_path = {f_path}, tmp_path = {tmp_path}")
                os.rename(f_path, tmp_path)
        cnt_cls = 0
        for root, _, files in os.walk(data_path):
            for f in files:
                cnt_cls += 1
                f_path = os.path.join(root, f)
                new_path = os.path.join(root, f"{cls}{cnt_cls}.jpg")
                print(f"f_path = {f_path}, tmp_path = {new_path}")
                os.rename(f_path, new_path)


def construct_ann_file():
    datasets = ["train", "val"]
    for dsets in datasets:
        train_cats_path = "./data/train/class_cats"
        train_dogs_path = "./data/train/class_dogs"
        val_cats_path = "./data/val/class_cats"
        val_dogs_path = "./data/val/class_dogs"
        train_ann_file = "./data/train.txt"
        val_ann_file = "./data/val.txt"
        cats_path = ""
        dogs_path = ""
        ann_file = ""

        if dsets.endswith("train"):
            cats_path = train_cats_path
            dogs_path = train_dogs_path
            ann_file = train_ann_file
        elif dsets.endswith("val"):
            cats_path = val_cats_path
            dogs_path = val_dogs_path
            ann_file = val_ann_file
            pass

        if os.path.exists(ann_file):
            os.remove(ann_file)

        clses = ["cats", "dogs"]
        d_path = ""
        cls_number = 0
        for cls in clses:
            if cls.endswith("cats"):
                d_path = cats_path
                cls_number = 0
                pass
            elif cls.endswith("dogs"):
                d_path = dogs_path
                cls_number = 1
                pass
            with open(ann_file, "a") as out_stream:
                for root, _, files in os.walk(d_path):
                    files = sorted(files, key=sort_by_number)
                    for f in files:
                        out_stream.write(f"{os.path.join(root, f).removeprefix(f'./data/{dsets}/')} {cls_number}\n")


def sort_by_number(path):
    # 通过正则表达式找到路径中的数字部分
    match = re.search(r'\d+', path)
    if match:
        return int(match.group())
    return path

if __name__ == '__main__':
    # arrange_train()
    # construct_val()
    # re_arrange_train()
    construct_ann_file()
