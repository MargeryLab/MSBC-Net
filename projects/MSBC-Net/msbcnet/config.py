# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_msbcnet_config(cfg):
    """
    Add config for MSBC-Net.
    """
    cfg.MODEL.MSBCNet = CN()
    cfg.MODEL.MSBCNet.NUM_CLASSES = 2
    cfg.MODEL.MSBCNet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.MSBCNet.NHEADS = 8
    cfg.MODEL.MSBCNet.DROPOUT = 0.0
    cfg.MODEL.MSBCNet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MSBCNet.ACTIVATION = 'relu'
    cfg.MODEL.MSBCNet.HIDDEN_DIM = 256
    cfg.MODEL.MSBCNet.NUM_CLS = 1
    cfg.MODEL.MSBCNet.NUM_REG = 3
    cfg.MODEL.MSBCNet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.MSBCNet.NUM_DYNAMIC = 2
    cfg.MODEL.MSBCNet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.MSBCNet.CLASS_WEIGHT = 2.0
    cfg.MODEL.MSBCNet.GIOU_WEIGHT = 2.0
    cfg.MODEL.MSBCNet.L1_WEIGHT = 5.0
    cfg.MODEL.MSBCNet.DEEP_SUPERVISION = True
    cfg.MODEL.MSBCNet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.MSBCNet.USE_FOCAL = True
    cfg.MODEL.MSBCNet.ALPHA = 0.25
    cfg.MODEL.MSBCNet.GAMMA = 2.0
    cfg.MODEL.MSBCNet.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    cfg.MODEL.BOUNDARY_MASK_HEAD = CN()
    cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV = 2
    cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_TYPE = 'ROIAlignV2'
