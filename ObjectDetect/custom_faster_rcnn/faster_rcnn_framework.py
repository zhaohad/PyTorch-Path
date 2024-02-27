from collections import OrderedDict

import torch
from torch import nn, Tensor

from ObjectDetect.custom_faster_rcnn.image_list import ImageList
from ObjectDetect.custom_faster_rcnn.rpn_function import RegionProposalNetwork, RPNHead, AnchorsGenerator
from ObjectDetect.custom_faster_rcnn.transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform, *args, **kwargs):
        super(FasterRCNNBase, self).__init__(*args, **kwargs)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]])\
            -> (dict[str, Tensor], list[dict[str, Tensor]]):
        image_trans: ImageList
        target_trans: list[dict[str, Tensor]]
        image_trans, target_trans = self.transform(images, targets)

        features = self.backbone(image_trans.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(image_trans, features, target_trans)


class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, roi_heads):
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 800
        max_size = 1333

        # 定义整个RPN框架
        rpn_anchor_generator = None
        rpn_head = None
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn_nms_thresh = 0.7,  # rpn中进行nms处理时使用的iou阈值
        rpn_score_thresh = 0.0,

        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            # TODO what's this
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
