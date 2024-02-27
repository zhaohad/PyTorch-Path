import math

import torch
import torchvision
from torch import Tensor
from torch import nn

from ObjectDetect.custom_faster_rcnn.image_list import ImageList
from datasets.data_presenter import DataPresenter


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        # if not isinstance(min_size, (list, tuple)):
        #     min_size = (min_size,)
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差
        self.presenter = DataPresenter()

    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]]) -> (ImageList, list[dict[str, Tensor]]):
        images = list(images)
        targets = list(targets)
        for i in range(len(images)):
            image = images[i]
            target = targets[i]

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))

            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list: list[tuple[int, int]] = []

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_sizes = [img.shape[-2:] for img in images]

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets

    def resize(self, image: Tensor, target: dict[str, Tensor]) -> [Tensor, dict[str, Tensor]]:
        h, w = image.shape[-2:]
        target = dict(target)
        # if self.training:
        #     size = float(self.torch_choice(self.min_size))
        #     assert False
        # else:
        #     size = float(self.min_size[-1])
        size = self.min_size

        if torchvision._is_tracing():
            assert False
        else:
            image = self._resize_image(image, size, float(self.max_size))
            pass

        box = target["boxes"]
        box = self.resize_boxes(box, [h, w], image.shape[-2:])
        target["boxes"] = box
        return image, target

    # def torch_choice(self, k):
    #     index = int(torch.empty(1).uniform_(0., float(len(k))).item())
    #     return k[index]

    def max_by_axis(self, the_list: list[list[int]]) -> list[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes


    def batch_images(self, images: list[Tensor], size_divisible: int = 32) -> Tensor:
        """
        把不同大小的image，提取batch中的最大size，最大size改成 size_divisible 的整数倍，
        按照这个数值构造矩形的Tensor，并将图片左上角对其复制，多余位置用0填充
        :param images:
        :param size_divisible:
        :return:
        """

        # if torchvision._is_tracing():
        #    return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size

        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 左上角对齐
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def _resize_image(self, image: Tensor, self_min_size: int, self_max_size: int) -> Tensor:
        img_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(img_shape))
        max_size = float(torch.max(img_shape))
        scale_factor = self_min_size / min_size

        if max_size * scale_factor > self_max_size:
            scale_factor = self_max_size / max_size

        # interpolate利用插值的方法缩放图片
        # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
        # bilinear只支持4D Tensor
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
            align_corners=False)[0]
        return image

    def resize_boxes(self, boxes: Tensor, original_size: list[int], new_size: list[int]) -> Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratios_height, ratios_width = ratios
        x_min, y_min, x_max, y_max = boxes.unbind(1)  # 对'1'维度进行拆分
        x_min = x_min * ratios_width
        x_max = x_max * ratios_width
        y_min = y_min * ratios_height
        y_max = y_max * ratios_height
        return torch.stack((x_min, y_min, x_max, y_max), dim=1)

