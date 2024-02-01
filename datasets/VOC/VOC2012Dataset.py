
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

import torch
from PIL import Image
from torch.utils.data import Dataset

MAIN_PATH = "ImageSets/Main"
MAIN_TRAIN = "train.txt"
ANNOTATIONS_PATH = "Annotations"
IMAGE_PATH = "JPEGImages"
MAIN_VAL = "val.txt"

g_classes_map: dict[str, int] = {}


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

    def __len__(self):
        return len(self._get_data_list())

    def __getitem__(self, item):
        data = self._get_data_list()[item]

        boxes = [[obj.bndbox.xmin, obj.bndbox.ymin, obj.bndbox.xmax, obj.bndbox.ymax] for obj in data.objects]
        labels = [g_classes_map[obj.name] for obj in data.objects]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = [obj.difficult if obj.difficult is not None else 0 for obj in data.objects]

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([data.img_id]),
            'area': area,
            'iscrowd': torch.as_tensor(is_crowd, dtype=torch.int64)
        }
        image = Image.open(data.img_path)
        return image, target

    def get_height_and_width(self, item):
        data = self._get_data_list()[item]
        return data.size.width, data.size.height


    def resolve_main(self, dir_path: str):
        train_file = os.path.join(dir_path, MAIN_TRAIN)
        self.resolve_main_file(train_file)
        val_file = os.path.join(dir_path, MAIN_VAL)
        self.resolve_main_file(val_file)

    def _get_data_list(self):
        data_list = {}
        if self.train_val.__eq__('train'):
            data_list = self.train
        elif self.train_val.__eq__('val'):
            data_list = self.val
        return data_list


    def resolve_main_file(self, file_path: str):
        if file_path.endswith(MAIN_TRAIN):
            data_list = self.train
        else:
            data_list = self.val
        with open(file_path, "r") as inF:
            for line in inF:
                data = RawData(image_id=len(data_list), img_dir=self.img_dir, ann_dir=self.ann_dir, name=line.strip())
                data_list.append(data)

    @staticmethod
    def collate_fn(batch):
        # print(f"type(batch) = {type(batch)}, len(batch) = {len(batch)}, "
        #       f"type(batch[0]) = {type(batch[0])}, len(batch[0]) = {len(batch[0])}, "
        #       f"type[batch[0][0]] = {type(batch[0][0])}, type[batch[0][1]] = {type(batch[0][1])}")
        # print(f"tuple(zip(*batch)) = {tuple(zip(*batch))}")
        # print(f"len(tuple(zip(*batch))) = {len(tuple(zip(*batch)))}")
        # print(f"zip(*batch) = {zip(*batch)}")
        return tuple(zip(*batch))


class RawData:
    def __init__(self, image_id: int, img_dir: str, ann_dir: str, name: str):
        self.img_id = image_id
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
                self._register_class(self.name)
            if child.tag == "pose":
                self.pose = child.text
            if child.tag == "truncated":
                self.truncated = child.text
            if child.tag == "occluded":
                self.occluded = child.text
            if child.tag == "difficult":
                self.difficult = int(child.text)
            if child.tag == "bndbox":
                self.bndbox = BndBox(child)

    def _register_class(self, cls: str):
        global g_classes_map
        if cls not in g_classes_map:
            g_classes_map[cls] = len(g_classes_map) + 1

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class BndBox:

    def __init__(self, item: Element):
        for child in item:
            if child.tag == "xmin":
                self.xmin = float(child.text)
            elif child.tag == "ymin":
                self.ymin = float(child.text)
            elif child.tag == "xmax":
                self.xmax = float(child.text)
            elif child.tag == "ymax":
                self.ymax = float(child.text)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
