# Copyright (c) wondervictor. All Rights Reserved
import os
from typing import List

import cv2
import fvcore.nn.weight_init as weight_init
import imageio
import torch
from mmcv.ops import ModulatedDeformConv2dPack
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            ModulatedDeformConv2dPack(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            ModulatedDeformConv2dPack(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)#（16,1536,14,14）
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def dice_loss_func(input, target):
    smooth = 1e-8
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def boundary_loss_func(boundary_logits, gtmasks):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    # 可视化
    boundary_logits_arr = torch.sigmoid(boundary_logits).detach().cpu().numpy()
    boundary_logits_arr_0 = boundary_logits_arr[0, 0, :, :] * 255
    cv2.imwrite('boundary_logits_0.png', boundary_logits_arr_0)

    gtmasks_arr = gtmasks.detach().cpu().numpy()
    gtmasks_arr_0 = gtmasks_arr[0, :, :] * 255
    cv2.imwrite('gt_mask.png', gtmasks_arr_0)

    bt_arr = boundary_targets.detach().cpu().numpy()
    bt_arr_0 = bt_arr[0, 0, :, :]
    bt_arr_0 = bt_arr_0 * 255
    cv2.imwrite('boundary_target.png', bt_arr_0)

    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        boundary_targets = F.interpolate(
            boundary_targets, boundary_logits.shape[2:], mode='nearest')

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)
    return bce_loss + dice_loss

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def boundary_preserving_mask_loss(
                                  pred_mask_logits,
                                  pred_boundary_logits,
                                  instances,
                                  vis_period=100):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor.detach(), mask_side_len
        ).to(device=pred_mask_logits.device)  # （2,28,28）
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)  # (16,28,28）

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)  # （0~15）
        gt_classes = cat(gt_classes, dim=0)  # 16（2,2,.....）
        pred_mask_logits = pred_mask_logits[indices, gt_classes]  # 取了25個28*28，只取【0，1，2】中对应真值的28*28
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)  # [6,28,28]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask/accuracy", mask_accuracy)
    storage.put_scalar("mask/false_positive", false_positive)
    storage.put_scalar("mask/false_negative", false_negative)

    pred_masks_t = pred_mask_logits.sigmoid()
    pred_masks_arr = pred_masks_t.detach().cpu().numpy()
    pred_masks_arr_0 = pred_masks_arr[0, :, :] * 255
    cv2.imwrite('pred_masks0.png', pred_masks_arr_0)

    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()

        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    boundary_loss = boundary_loss_func(pred_boundary_logits, gt_masks)
    return mask_loss, boundary_loss


@ROI_MASK_HEAD_REGISTRY.register()
class BoundaryMaskHead(nn.Module):

    def __init__(self, cfg, bilinear=False):
        super(BoundaryMaskHead, self).__init__()
        self.n_channels = 256
        num_classes = cfg.MODEL.MSBCNet.NUM_CLASSES
        self.bilinear = bilinear

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM

        self.inc = DoubleConv(self.n_channels, 256)
        self.down1 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down2 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.outc = OutConv(256, num_classes)

        self.bfc1 = BasicConv2d(256, 512, kernel_size=1, padding=0)
        self.bfc2 = BasicConv2d(512, 1024, kernel_size=1, padding=0)

        self.boundary_to_mask1 = Conv2d(
            256, 512,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, 512),
            activation=F.relu
        )

        self.boundary_to_mask2 = Conv2d(
            512, 1024,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, 1024),
            activation=F.relu
        )

        self.mask_to_boundary = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.bft1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv2d(
            512, 256,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, 256),
            activation=F.relu
        )
        )

        cur_channels = 256

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )

        self.boundary_deconv = ConvTranspose2d(
            512, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, mask_features, boundary_features, instances):    #（27,256,28,28）,（27,256,28,28）

        x1 = self.inc(mask_features)    #（27,256,28,28）
        boundary_features = boundary_features + self.mask_to_boundary(x1)   #27,256,28,28
        boundary_features = self.inc(boundary_features) #27,256,28,28

        x2 = self.down1(x1) #（27，512，14，14)
        # x2 = F.interpolate(self.boundary_to_mask1(boundary_features),scale_factor=1/2, mode='bilinear', align_corners=True) + x2 #插值合适还是下采样卷积合适
        boundary_features_tmp = self.boundary_to_mask1(boundary_features) #27,512,14,14
        x2 = boundary_features_tmp + x2  #27,512,14,14
        x3 = self.down2(x2) #(27,1024,7,7）
        x3 = self.boundary_to_mask2(boundary_features_tmp) + x3 #27,1024,7,7

        x = self.up1(x3, x2)    #（27,512,14,14）
        x = self.up2(x, x1) #（27,256,28,28）
        mask_logits = self.outc(x)  #（27,2,28,28）

        boundary_logits = self.outc(boundary_features)

        if self.training:
            loss_mask, loss_boundary = boundary_preserving_mask_loss(
                mask_logits, boundary_logits, instances)
            return {"loss_mask": loss_mask,
                    "loss_boundary": loss_boundary}
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances