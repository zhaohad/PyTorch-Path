import torch
import matplotlib.pyplot as plt

import os

import torch
from torch.utils.data import DataLoader

from ObjectDetect.custom_faster_rcnn.backbone.mobilenetv2_model import MobileNetV2
from ObjectDetect.custom_faster_rcnn.faster_rcnn_framework import FasterRCNN
from ObjectDetect.custom_faster_rcnn.image_list import ImageList
from ObjectDetect.custom_faster_rcnn.transform import GeneralizedRCNNTransform
from datasets.VOC.VOC2012Dataset import VOC2012Dataset
from datasets.VOC.voc_data_presenter import VOCDataPresenter
from torchvision import datasets, transforms

from datasets.data_presenter import DataPresenter

VOC_DATA_ROOT_PATH = "data/VOCdevkit/VOC2012"
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model():
    # TODO how to build the backbone
    backbone = MobileNetV2(weights_path="pretrain_models/mobilenet/mobilenet_v2-b0353104.pth").features
    backbone.out_channels = 1280

    # TODO build model
    faster_rcnn_model = FasterRCNN(backbone=backbone, roi_heads=None)

    # TODO use gpu
    # faster_rcnn_model.to(DEVICE)

    return faster_rcnn_model


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    voc_dataset = VOC2012Dataset(root_dir=VOC_DATA_ROOT_PATH, train_val="train", transform=transform)
    presenter = VOCDataPresenter()
    # presenter.show_class(voc_dataset, 'dog')

    faster_rcnn = create_model()

    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = DataLoader(dataset=voc_dataset,
                                   # batch_sampler=train_batch_sampler,
                                   batch_size=8,
                                   # batch_size=1,
                                   pin_memory=True,
                                   # num_workers=nw,
                                   num_workers=1,
                                   collate_fn=voc_dataset.collate_fn)

    i: int = 0
    for batch, [images, targets] in enumerate(train_data_loader):
        faster_rcnn.train()
        faster_rcnn(images, targets)
        i += 1
        if i >= 2:
            break


if __name__ == "__main__":
    main()
