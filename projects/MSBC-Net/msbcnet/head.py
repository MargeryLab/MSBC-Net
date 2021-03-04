# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MSBC-Net Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, Instances, pairwise_iou, PolygonMasks
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.mask_head import build_mask_head

from .deform_attention import MSDeformAttn

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def add_ground_truth_to_proposals_single_image(gt_boxes, gt_classes, gt_masks, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.gt_boxes = gt_boxes
    gt_proposal.gt_classes = gt_classes
    gt_proposal.gt_masks = gt_masks
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals

def add_ground_truth_to_proposals(gt_instances, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    gt_masks = []
    gt_boxes = [x.gt_boxes for x in gt_instances]
    gt_classes = [x.gt_classes for x in gt_instances]
    for x in gt_instances:
        try:
            gt_masks.append(x.gt_masks)
        except:
            gt_masks.append([])
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    assert len(proposals) == len(gt_classes)
    if len(proposals) == 0:
        return proposals

    proposals_add_gt = [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, gt_classes_i, gt_masks_i, proposals_i)
        for gt_boxes_i, gt_classes_i, gt_masks_i, proposals_i in zip(gt_boxes, gt_classes, gt_masks, proposals)
    ]
    return proposals_add_gt

def select_foreground_proposals(proposals: List[Instances], bg_label: int):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)

    fg_proposals = []
    for proposals_per_image in proposals:
        iter = 0
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        iter += 1

    return fg_proposals

class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.mask_on = cfg.MODEL.MASK_ON
        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        mask_pooler = self._init_mask_pooler(cfg, roi_input_shape)
        boundary_pooler = self._init_boundary_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        self.mask_pooler = mask_pooler
        self.boundary_pooler = boundary_pooler

        # Build heads.
        num_classes = cfg.MODEL.MSBCNet.NUM_CLASSES
        d_model = cfg.MODEL.MSBCNet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.MSBCNet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.MSBCNet.NHEADS
        dropout = cfg.MODEL.MSBCNet.DROPOUT
        activation = cfg.MODEL.MSBCNet.ACTIVATION
        num_heads = cfg.MODEL.MSBCNet.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = cfg.MODEL.MSBCNet.DEEP_SUPERVISION

        # Init parameters.
        self.use_focal = cfg.MODEL.MSBCNet.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.MSBCNet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    @staticmethod
    def _init_mask_pooler(cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        # in_channels = [input_shape[f].channels for f in in_features][0]

        mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return mask_pooler

    @staticmethod
    def _init_boundary_pooler(cfg, input_shape):
        if not cfg.MODEL.BOUNDARY_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES
        pooler_resolution = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_TYPE
        # fmt: on

        # in_channels = [input_shape[f].channels for f in in_features][0]

        boundary_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return boundary_pooler

    @torch.no_grad()
    def label_proposals(self, proposals, gt_instances: List[Instances]):
        """
        Args:
            proposals : proposals for each image.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """

        gt_boxes = [x.gt_boxes for x in gt_instances]  # 32个box
        image_sizes = [x.image_size for x in gt_instances]  # 32个size

        del gt_instances
        N, nr_boxes = proposals.shape[:2]  # 16，100
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(proposals[b]))

        results_pn = []
        for image_size_i, gt_boxes_i in \
                zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            iter = 0
            res = Instances(image_size_i)
            res.proposal_boxes = proposal_boxes[iter]
            results_pn.append(res)
            iter += 1

        return results_pn

    @torch.no_grad()
    def label_proposals_test(self, bboxes, size):
        image_sizes = [x for x in size]
        proposal_boxes = list()
        if bboxes.ndim == 2:
            proposal_boxes.append(Boxes(bboxes))
        else:
            N, nr_boxes = bboxes.shape[:2]
            for b in range(N):
                proposal_boxes.append(Boxes(bboxes[b]))

        results_pn = []
        for image_size_i, boxes_i in \
                zip(image_sizes, proposal_boxes):
            """
            image_size_i: (h, w) for the i-th image
            """
            res = Instances(image_size_i)
            res.pred_boxes = boxes_i
            results_pn.append(res)

        return results_pn

    def forward(self, src, init_bboxes, init_features, gt_instances_or_size):
        inter_class_logits = []
        inter_pred_bboxes = []
        inter_loss_bmask = []
        inter_eval_pred = []

        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        bs = len(features[0])
        bboxes = init_bboxes

        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()

        if self.training:
            proposals_ins = self.label_proposals(bboxes, gt_instances_or_size)
            for rcnn_head in self.head_series:
                if self.mask_on:
                    class_logits, pred_bboxes, proposal_features, loss_bmask_head = \
                        rcnn_head(src, bboxes, proposal_features, proposals_ins, self.box_pooler, self.mask_pooler,
                                  self.boundary_pooler, gt_instances_or_size)

                    if self.return_intermediate:
                        inter_class_logits.append(class_logits)
                        inter_pred_bboxes.append(pred_bboxes)
                        inter_loss_bmask.append(loss_bmask_head)
                else:
                    class_logits, pred_bboxes, proposal_features = \
                        rcnn_head(src, bboxes, proposal_features, proposals_ins, self.box_pooler, self.mask_pooler,
                                  self.boundary_pooler, gt_instances_or_size)

                    if self.return_intermediate:
                        inter_class_logits.append(class_logits)
                        inter_pred_bboxes.append(pred_bboxes)
                bboxes = pred_bboxes.detach()
        else:
            for rcnn_head in self.head_series:
                proposals_ins = self.label_proposals_test(bboxes, gt_instances_or_size)
                if self.mask_on:
                    eval_res, pred_bboxes, proposal_features = \
                        rcnn_head(src, bboxes, proposal_features, proposals_ins, self.box_pooler,
                                  self.mask_pooler,
                                  self.boundary_pooler, gt_instances_or_size)
                    if self.return_intermediate:
                        inter_eval_pred.append(eval_res)
                else:
                    class_logits, pred_bboxes, proposal_features = \
                        rcnn_head(src, bboxes, proposal_features, proposals_ins, self.box_pooler,
                                  self.mask_pooler,
                                  self.boundary_pooler, gt_instances_or_size)
                    if self.return_intermediate:
                        inter_class_logits.append(class_logits)
                        inter_pred_bboxes.append(pred_bboxes)
                bboxes = pred_bboxes.detach()

        if self.mask_on:
            if self.training:
                if self.return_intermediate:
                    return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), inter_loss_bmask

                return class_logits[None], pred_bboxes[None], inter_loss_bmask
            else:
                if self.return_intermediate:
                    return inter_eval_pred
        else:
            if self.return_intermediate:
                return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

            return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.cfg = cfg
        self.d_model = d_model
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_classes = cfg.MODEL.MSBCNet.NUM_CLASSES
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.proposal_append_gt = True
        self.test_score_thresh = 0.05
        self.test_nms_thresh = 0.5
        self.test_topk_per_image = 10
        self.score_thresh = 0.05

        self.proposal_matcher = Matcher([0.5], [0, 1], allow_low_quality_matches=False)


        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MSDeformAttn(d_model, 1, nhead, n_points=4)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.MSBCNet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.MSBCNet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.MSBCNet.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        self.box_pool_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.mask_pool_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.boundary_pool_resolution = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION
        self.mask_head = build_mask_head(cfg)

    def _sample_proposals(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            # gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # sampled_fg_idxs = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
        # sampled_bg_idxs = nonzero_tuple(gt_classes == self.num_classes)[0]
        #
        # s1 = torch.tensor([2],dtype=torch.int64)
        # s2 = torch.tensor([0,1,3,4,5,6],dtype=torch.int64)
        # s = torch.cat([s1,s2],dim=0)
        # sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        # return sampled_idxs, gt_classes[sampled_idxs]
        return gt_classes

    @torch.no_grad()
    def label_proposals_gt_classes(self, proposals: List[Instances], gt_instances: List[Instances]):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, gt_instances):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[matched_idxs])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(matched_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def predict_probs(self, scores, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        num_inst_per_image = [len(p) for p in proposals]
        # probss = F.softmax(scores, dim=-1)
        probs = torch.sigmoid(scores)
        return probs.split(num_inst_per_image, dim=0)

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, src, bboxes, pro_features, proposals_ins, box_pooler, mask_pooler, boundary_pooler, gt_instances):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """
        N, nr_boxes = bboxes.shape[:2]
        box_features = [src[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES]
        mask_features = [src[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES]  # （0,1,2,3）
        boundary_features = [src[f] for f in self.cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = box_pooler(box_features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.(1,1200,256)
        # pro_features = pro_features.view(N, nr_boxes, self.d_model)
        # spatial_shapes = []
        # if pro_features.shape[1] == 100:
        #     spatial_shapes.append((10, 10))
        # else:
        #     spatial_shapes.append((20, 15))
        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device='cuda')
        # level_start_index = torch.tensor([0], device='cuda')
        # reference_points = self.get_reference_points(spatial_shapes, device='cuda')
        # pro_features2 = self.self_attn(pro_features, reference_points, pro_features, spatial_shapes, level_start_index).permute(1,0,2)
        # pro_features = pro_features.permute(1,0,2) + self.dropout1(pro_features2)
        # pro_features = self.norm1(pro_features)

        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        if self.mask_on:
            if self.training:
                class_logits = class_logits.view(N, nr_boxes, -1)
                pred_bboxes = pred_bboxes.view(N, nr_boxes, -1)  # tensor（16,100,4）
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        for proposals_per_image, pred_boxes_per_image in zip(proposals_ins, pred_bboxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

                proposals_with_gt = self.label_proposals_gt_classes(proposals_ins, gt_instances)
                if self.proposal_append_gt:
                    proposals_with_gt = add_ground_truth_to_proposals(gt_instances, proposals_with_gt)

                proposals_fg_ins = select_foreground_proposals(proposals_with_gt, self.num_classes)
                proposals_fg = [x.proposal_boxes for x in proposals_fg_ins]
                roi_mask_features = mask_pooler(mask_features, proposals_fg)  # (1600,256,14,14)
                # roi_mask_features = roi_mask_features.view(roi_mask_features.shape[0], self.d_model, -1).permute(2, 0,
                #                                                                                                  1)  # (196,1600,256)
                roi_boundary_features = boundary_pooler(boundary_features, proposals_fg)  # (1600,256,28,28)
                loss_bmask_head = self.mask_head(roi_mask_features, roi_boundary_features, proposals_fg_ins)
                return class_logits, pred_bboxes, obj_features, loss_bmask_head
            else:
                pred_bboxes = pred_bboxes.view(N, nr_boxes, -1)  # tensor（16,100,4）

                scores = self.predict_probs(class_logits, proposals_ins)

                pred_ins = []
                predi_bboxes = []
                for proposals_per_image, pred_boxes_per_image, scores_per_image in zip(proposals_ins, pred_bboxes,
                                                                                       scores):
                    filter_mask = scores_per_image > self.score_thresh
                    filter_inds = filter_mask.nonzero()
                    filter_inds_tmp = torch.zeros(2,2).cuda()
                    if filter_inds.shape[0] == 0:
                        filter_inds_tmp = filter_inds
                        filter_mask = scores_per_image > 0.001
                        filter_inds = filter_mask.nonzero()
                    scores_i = scores_per_image[filter_mask]
                    pred_boxes_per_img = pred_boxes_per_image[filter_inds[:, 0]]
                    per_ins = Instances(proposals_per_image.image_size)
                    per_ins.pred_classes = filter_inds[:, 1]
                    per_ins.scores = scores_i
                    per_ins.pred_boxes = Boxes(pred_boxes_per_img)
                    pred_ins.append(per_ins)
                    predi_bboxes.append(Boxes(pred_boxes_per_img))

                    if filter_inds_tmp.shape[0] == 0:
                        for i in range(len(pred_ins)):
                            x = torch.zeros(pred_ins[i].pred_classes.shape[0], 1, 28, 28).cuda()
                            pred_ins[i].pred_masks = x
                        return pred_ins, pred_bboxes, obj_features

                roi_mask_features = mask_pooler(mask_features, predi_bboxes)
                roi_boundary_features = boundary_pooler(boundary_features, predi_bboxes)
                result = self.mask_head(roi_mask_features, roi_boundary_features, pred_ins)
                return result, pred_bboxes, obj_features
        else:
            return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.MSBCNet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.MSBCNet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.MSBCNet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")