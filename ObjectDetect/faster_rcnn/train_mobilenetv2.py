import os
import datetime
import torch
import torchvision
from torchvision.models.detection import FasterRCNN


def create_model(num_classes):
    model = FasterRCNN()
