import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ObjectDetect.custom_faster_rcnn import det_utils
from ObjectDetect.custom_faster_rcnn import boxes as box_ops
from ObjectDetect.custom_faster_rcnn.image_list import ImageList


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C, H, W)
    print(f"layer.shape 1 = {layer.shape}")
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    print(f"layer.shape 2 = {layer.shape}")
    layer = layer.reshape(N, -1, C)
    print(f"layer.shape 3 = {layer.shape}")
    return layer


def concat_box_prediction_layers(box_cls: list[Tensor], box_regression: list[Tensor]) -> (Tensor, Tensor):
    """
    对box_cla和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]
    Args:
        box_cls: 每个预测特征层上的预测目标概率
        box_regression: 每个预测特征层上的预测目标bboxes regression参数

    Returns:

    """
    box_cls_flattened = []
    box_regression_flattened = []
    print(f"box_cls[0].shape = {box_cls[0].shape}, box_regression[0].shape = {box_regression[0].shape}")

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A
        print(f"N = {N}, AxC = {AxC}, H = {H}, W = {W}, Ax4 = {Ax4}, A = {A}, C = {C}")

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    print(f"box_cls.shape = {box_cls.shape}, box_regression.shape = {box_regression.shape}")
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.


    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            if False and torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals: Tensor, objectness: Tensor, image_shapes: list[(int, int)], num_anchors_per_level: list[int]) -> (list[Tensor, list[Tensor]]):
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标
            objectness: 预测的目标概率
            image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目

        Returns:

        """
        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def forward(self, images: ImageList, features: dict[str, Tensor], targets: list[dict[str, Tensor]] = None)\
            -> (list[Tensor], dict[str, Tensor]):
        features = list(features.values())  # list[Tensor(b, c, w, h)]

        objectness, pred_bbox_deltas = self.head(features)

        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        for o in objectness:
            print(f"o.shape = {o.shape}, o[0].shape = {o[0].shape}")
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        print(f"num_anchors_per_level_shape_tensors = {num_anchors_per_level_shape_tensors}, num_anchors_per_level = {num_anchors_per_level}")

        # objectness 是Anchor的数量，每个Anchor要取一个cls分类，pred_bbox_deltas是Anchor数量乘以4，也就是预测框的4个值
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        print(f"proposals.shape = {proposals.shape}")

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        return None, None


class RPNHead(torch.nn.Module):

    def __init__(self, in_channels, num_anchors):
        print(f"RPNHead __init__ num_anchors = {num_anchors}")
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, features: list[Tensor]) -> [list[Tensor], list[Tensor]]:
        logits = []
        bbox_reg = []
        for i, feature in enumerate(features):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class AnchorsGenerator(torch.nn.Module):

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        # TODO init
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def generate_anchors(self, scales: list[int], aspect_ratios: list[float],
                         dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device('cpu')) -> Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios
        # w_ratios[:, None]： 把w_ratios第一维的所有元素提出来然后组成一个二维数组
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()  # round 四舍五入

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors
        print(f"self.cell_anchors = {self.cell_anchors}")

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: list[list[int]], strides: list[list[Tensor]]) -> list[Tensor]:
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            print(f"size = {size}, stride = {stride}, base_anchors = {base_anchors}")
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的x坐标(列)
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            print(f"shifts_x = {shifts_x}\nshifts_y = {shifts_y}")

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            print(f"shift_x.shape = {shift_x.shape}, shift_y.shape = {shift_x.shape}, shifts.shape = {shifts.shape}")

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            print(f"shifts_anchor.shape = {shifts_anchor.shape}")
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes: list[list[int]], strides: list[list[Tensor]]) -> list[Tensor]:
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        print(f"key = {key}")
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list: ImageList, feature_maps: list[Tensor]) -> list[Tensor]:
        # TODO feature的h w是如何确定的？
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]

        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        self.set_cell_anchors(dtype, device)
        print(f"grid_sizes = {grid_sizes}, image_size = {image_size}, strides = {strides}")
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(list[list[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        print(f"len[anchors] = {len(anchors)}, anchors[0].shape = {anchors[0].shape}")
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors

