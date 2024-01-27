
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from torch.utils.data import Dataset

MAIN_PATH = "ImageSets/Main"
MAIN_TRAIN = "train.txt"
ANNOTATIONS_PATH = "Annotations"
IMAGE_PATH = "JPEGImages"
MAIN_VAL = "val.txt"


class VOC2012Dataset(Dataset):

    def __init__(self, root_dir: str, train_val: str = 'ALL'):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, IMAGE_PATH)
        self.ann_dir = os.path.join(self.root_dir, ANNOTATIONS_PATH)

        self.train_val = train_val
        self.main_path = os.path.join(root_dir, MAIN_PATH)

        self.train: list[RawData] = []
        self.val: list[RawData] = []

        self.resolve_main(self.main_path)
        print(len(self.train))
        print(len(self.val))

    def resolve_main(self, dir_path: str):
        train_file = os.path.join(dir_path, MAIN_TRAIN)
        self.resolve_main_file(train_file)
        val_file = os.path.join(dir_path, MAIN_VAL)
        self.resolve_main_file(val_file)

    def resolve_main_file(self, file_path: str):
        if file_path.endswith(MAIN_TRAIN):
            data_list = self.train
        else:
            data_list = self.val
        with open(file_path, "r") as inF:
            for line in inF:
                data_list.append(RawData(img_dir=self.img_dir, ann_dir=self.ann_dir, name=line.strip()))


class RawData:

    def __init__(self, img_dir: str, ann_dir: str, name: str):
        self.img_path = os.path.join(img_dir, f"{name}.jpg")
        self.ann_path = os.path.join(ann_dir, f"{name}.xml")
        assert os.path.exists(self.img_path)
        assert os.path.exists(self.ann_path)

        tree = ET.parse(self.ann_path)
        root = tree.getroot()
        self.objects: list[Object] = []

        for child in root:
            if child.tag == "size":
                self.size: Size = Size(child)
            if child.tag == "object":
                self.objects.append(Object(child))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class Size:

    def __init__(self, item: Element):
        for child in item:
            if child.tag == "width":
                self.width = child.text
            elif child.tag == "height":
                self.height = child.text
            elif child.tag == "depth":
                self.depth = child.text

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class Object:

    def __init__(self, item: Element):
        for child in item:
            if child.tag == "name":
                self.name = child.text
            if child.tag == "pose":
                self.pose = child.text
            if child.tag == "truncated":
                self.truncated = child.text
            if child.tag == "occluded":
                self.occluded = child.text
            if child.tag == "difficult":
                self.difficult = child.text
            if child.tag == "bndbox":
                self.bndbox = BndBox(child)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class BndBox:

    def __init__(self, item: Element):
        for child in item:
            if child.tag == "xmin":
                self.xmin = child.text
            elif child.tag == "ymin":
                self.ymin = child.text
            elif child.tag == "xmax":
                self.xmax = child.text
            elif child.tag == "ymax":
                self.ymax = child.text

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
