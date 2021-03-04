#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os, csv
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized)

__all__ = ["MSBCNet"]

def dice_coefficient(input, target):
    smooth = 1e-8
    intersection = (input * target).sum(1)
    dice = (2. * intersection + smooth) / (input.sum(1) + target.sum(1) + smooth)
    return dice.mean()

@META_ARCH_REGISTRY.register()
class MSBCNet(nn.Module):
    """
    Implement MSBC-Net
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.mask_on = cfg.MODEL.MASK_ON

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.MSBCNet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.MSBCNet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.MSBCNet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.MSBCNet.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.MSBCNet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.MSBCNet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.MSBCNet.L1_WEIGHT
        no_object_weight = cfg.MODEL.MSBCNet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.MSBCNet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.MSBCNet.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def adjust_box(self, box_t):
        if box_t.data[0][2] - box_t.data[0][0] < 0:
            box2 = box_t.data[0][2].clone()
            box_t.data[0][2] = box_t.data[0][0]
            box_t.data[0][0] = box2
        if box_t.data[0][3] - box_t.data[0][1] < 0:
            box3 = box_t.data[0][3].clone()
            box_t.data[0][3] = box_t.data[0][1]
            box_t.data[0][1] = box3
        return box_t

    def prepare_proposals(self, images_whwh):
        result = []
        res_100 = []
        proposal_boxes = self.init_proposal_boxes.weight.clone()  # （100,4）
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] % 1 * (
        images_whwh[:, None, :])  # torch.Size([16, 100, 4])乘以imge_size whwh[352,352,352,352]
        for i in range(proposal_boxes.shape[0]):
            res_100.clear()
            # threshold = [0, images_whwh[0].cpu().numpy()[0]]
            for j in range(proposal_boxes[i].shape[0]):
                box = proposal_boxes[i][j].reshape(1, 4)
                # _,box = self.init_one_box(box, threshold,images_whwh)
                box = self.adjust_box(box)
                res_100.append(box)
            res = torch.cat(res_100, dim=0).reshape(self.num_proposals, 4)
            result.append(res)

        result = torch.cat(tuple(result), dim=0).reshape(len(images_whwh), self.num_proposals, 4)
        return result

    def postprogress(self, results, batched_inputs, image_sizes):
        processed_results = []
        tumor_mask_root = './test_tumor_whole'
        wall_mask_root = './test_wall_whole'
        pred_tumor_path = './predictionTumor'
        pred_wall_path = './predictionWall'
        if not os.path.exists(pred_tumor_path):
            os.mkdir(pred_tumor_path)
        if not os.path.exists(pred_wall_path):
            os.mkdir(pred_wall_path)

        pred_tumor_masks = []
        pred_wall_masks = []
        gt_wall_masks = []
        gt_tumor_masks = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})

            if self.mask_on:
                img_name = input_per_image['file_name'].split('/')[-1]

                dice_path = 'diceLog.csv'
                readstyle = 'a+'
                if img_name in pd.read_csv(dice_path, usecols=['SubjectID']):
                    readstyle = "w+"

                if r.pred_classes.shape == torch.Size([0]) and \
                        (np.max(cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0)) == 0) and \
                        (np.max(cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0)) == 0): #没有检测出目标
                    with open(dice_path, readstyle, newline='') as file:
                        csv_file = csv.writer(file)
                        datas = [img_name, 1., 1., 1.]
                        csv_file.writerow(datas)
                    continue
                elif r.pred_classes.shape == torch.Size([0]) and \
                        (np.max(cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0)) == 0) and \
                        (np.max(cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0)) != 0):
                    with open(dice_path, readstyle, newline='') as file:
                        csv_file = csv.writer(file)
                        datas = [img_name, 1., 0, 0.5]
                        csv_file.writerow(datas)
                    continue
                elif r.pred_classes.shape == torch.Size([0]) and \
                        (np.max(cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0)) != 0) and \
                        (np.max(cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0)) == 0):
                    with open(dice_path, readstyle, newline='') as file:
                        csv_file = csv.writer(file)
                        datas = [img_name, 0, 1., 0.5]
                        csv_file.writerow(datas)
                    continue

                if r.pred_classes.min().cpu().numpy() == 0 and (    #兩個类都有
                        r.pred_classes.max().cpu().numpy() == self.num_classes - 1):
                    for i in range(self.num_classes):
                        pred_sum = np.zeros((r.pred_masks.shape[1], r.pred_masks.shape[2]), dtype=np.int32)
                        pred_classes = r.pred_classes
                        index = (pred_classes == i).nonzero()
                        index = index.reshape(len(index))
                        scores, idx = r.scores[index].sort(descending=True)
                        pred_masks = r.pred_masks[index][idx]  # （7，310，420）
                        if i == 0:
                            for j in range(pred_masks.shape[0]):
                                pred_sum += pred_masks[j].int().cpu().numpy()
                            pred_sum[pred_sum >= 2] = 1
                            assert pred_sum.max() <= 1
                            pred = (pred_sum * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(pred_tumor_path, img_name),pred)
                            pred_tumor_masks.append(torch.tensor(pred_sum,dtype=torch.float32,device='cuda'))
                            mask_path = os.path.join(tumor_mask_root, img_name)
                            mask_arr = cv2.imread(mask_path, flags=0)  # （320,410）
                            gt_tumor_masks.append(torch.from_numpy(mask_arr/255.).type(torch.float32).cuda())
                        else:
                            for j in range(pred_masks.shape[0]):
                                pred_sum += pred_masks[j].int().cpu().numpy()
                            pred_sum[pred_sum >= 2] = 1
                            assert pred_sum.max() <= 1
                            pred = (pred_sum * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(pred_wall_path, img_name),pred)
                            pred_wall_masks.append(torch.tensor(pred_sum,dtype=torch.float32,device='cuda'))
                            mask_path = os.path.join(wall_mask_root, img_name)
                            mask_arr = cv2.imread(mask_path, flags=0)
                            gt_wall_masks.append(torch.from_numpy(mask_arr/255.).type(torch.float32).cuda())
                else:
                    scores, idx = r.scores.sort(descending=True)
                    pred_masks = r.pred_masks[idx]  # （7，310，4）
                    pred_sum = np.zeros((r.pred_masks.shape[1], r.pred_masks.shape[2]), dtype=np.int32)
                    if r.pred_classes.max().item() == 0:
                        for i in range(pred_masks.shape[0]):
                            pred_sum += pred_masks[i].int().cpu().numpy()
                        pred_sum[pred_sum >= 2] = 1
                        assert pred_sum.max() <= 1
                        pred = (pred_sum * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(pred_tumor_path, img_name),pred)
                        pred_tumor_masks.append(torch.tensor(pred_sum,dtype=torch.float32,device='cuda'))
                        mask_path = os.path.join(tumor_mask_root, img_name)
                        mask_arr = cv2.imread(mask_path, flags=0)
                        gt_tumor_masks.append(torch.from_numpy(mask_arr/255.).type(torch.float32).cuda())
                    else:
                        for i in range(pred_masks.shape[0]):
                            pred_sum += pred_masks[i].int().cpu().numpy()
                        pred_sum[pred_sum >= 2] = 1
                        assert pred_sum.max() <= 1
                        pred = (pred_sum * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(pred_wall_path, img_name), pred)
                        pred_wall_masks.append(torch.tensor(pred_sum,dtype=torch.float32,device='cuda'))
                        mask_path = os.path.join(wall_mask_root, img_name)
                        mask_arr = cv2.imread(mask_path, flags=0)
                        gt_wall_masks.append(torch.from_numpy(mask_arr/255.).type(torch.float32).cuda())

                if pred_tumor_masks:
                    tumor_dice_mean = dice_coefficient(torch.cat(pred_tumor_masks, dim=1).view(len(pred_tumor_masks), -1),
                                                       torch.cat(gt_tumor_masks, dim=1).view(len(pred_tumor_masks), -1))
                elif cv2.imread(os.path.join(tumor_mask_root, img_name), flags=0).max() == 0:
                    tumor_dice_mean = torch.tensor(1.).to(self.device)
                else:
                    tumor_dice_mean = torch.tensor(0.).to(self.device)

                if pred_wall_masks:
                    wall_dice_mean = dice_coefficient(torch.cat(pred_wall_masks, dim=1).view(len(pred_wall_masks), -1),
                                                      torch.cat(gt_wall_masks, dim=1).view(len(pred_wall_masks), -1))
                elif cv2.imread(os.path.join(wall_mask_root, img_name), flags=0).max() == 0:
                    wall_dice_mean = torch.tensor(1.).to(self.device)
                else:
                    wall_dice_mean = torch.tensor(0.).to(self.device)
                dice_mean = (tumor_dice_mean + wall_dice_mean) / 2

                with open(dice_path, readstyle, newline='') as file:
                    csv_file = csv.writer(file)
                    datas = [img_name, tumor_dice_mean.item(), wall_dice_mean.item(), dice_mean.item()]
                    csv_file.writerow(datas)

        return processed_results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)

        # Prepare Proposals.
        # proposal_boxes = self.init_proposal_boxes.weight.clone()
        # proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        # proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        proposal_boxes = self.prepare_proposals(images_whwh)

        # Prediction.
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.mask_on:
                outputs_class, outputs_coord, loss_bmask_head = self.head(src, proposal_boxes, self.init_proposal_features.weight,gt_instances)
                output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'bmask_loss': loss_bmask_head[-1]}
                if self.deep_supervision:
                    output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'bmask_loss': c}  # a,(8，100，8)
                                             for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], loss_bmask_head[
                                                                                                        :-1])]
            else:
                outputs_class, outputs_coord = self.head(src, proposal_boxes, self.init_proposal_features.weight,
                                                         gt_instances)
                output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
                if self.deep_supervision:
                    output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            if self.mask_on:
                results = self.head(src, proposal_boxes, self.init_proposal_features.weight, images.image_sizes)
                results = results[-1]
            else:
                outputs_class, outputs_coord = self.head(src, proposal_boxes, self.init_proposal_features.weight,
                                                         images.image_sizes)
                output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = self.postprogress(results, batched_inputs, images.image_sizes)
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
