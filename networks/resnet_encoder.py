# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate as resize_tensor

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list


class ResnetEncoder(nn.Module):
    """
    Main class for SimVODIS.
    - SimVODIS extends the mask-rcnn model.
    - It implements the fourth and fifth branches from the features
    computed by the backbone module of mask-rcnn.
    - The fourth branch calculates the relative pose between two images.
    - The fifthe branch estimates the depth map of the center image.
    - Thus, SimVODIS produces
    1) object class (part of D: detection)
    2) bounding box (part of D: detection)
    3) segmentation mask (IS: instance segmentation)
    4) relative pose (VO: visual odometry)
    5) depth map
    """

    def __init__(
        self, cfg, pretrained_model_path, build_transform=False, joint_training=False
    ):
        super(ResnetEncoder, self).__init__()

        # basic properties
        self.cfg = cfg
        self.transforms = None
        if build_transform:
            self.transforms = self.build_transform()
        self.joint_training = joint_training
        self.device = torch.device(cfg.MODEL.DEVICE)

        # loading mask rcnn
        self.maskrcnn = build_detection_model(cfg)
        self.maskrcnn.eval()
        device = torch.device(cfg.MODEL.DEVICE)
        self.maskrcnn.to(device)
        self.checkpointer = DetectronCheckpointer(
            cfg, self.maskrcnn, save_dir='.'
        )
        _ = self.checkpointer.load(pretrained_model_path)

        if not self.joint_training:
            # freeze gradients for mask rcnn
            for param in self.maskrcnn.backbone.parameters():
                param.requires_grad = False
            for param in self.maskrcnn.rpn.parameters():
                param.requires_grad = False
            for param in self.maskrcnn.roi_heads.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Arguments:
            - image_reference (np.ndarray): reference image
            - image_before, image_after (np.ndarray): target image
            whose pose will be calculated

        Returns:
            - pose (list[Tensor]): the output from the fourth
            branch of SimVODIS. It contains the relative poses
            between neighboring image frames.
            - features (): the output from the mask-rcnn model.
            - predictions (list[BoxList]): the output from the
            mask-rcnn model. It returns list[BoxList] containing
            additional fields such as `scores`, `labels` and `mask`.
        """
        '''
        assert img_before.shape == img_ref.shape, ("reference and target in "
                                                   "different shape")

        if not self.training and self.transforms is not None:
            img_before = self.transforms(img_before)
            img_ref = self.transforms(img_ref)
            img_after = self.transforms(img_after)

        img_before = img_before.to(self.device)
        img_ref = img_ref.to(self.device)
        img_after = img_after.to(self.device)

        if img_ref.dim() == 3:
            img_before = img_before.unsqueeze(0)
            img_ref = img_ref.unsqueeze(0)
            img_after = img_after.unsqueeze(0)
        images = torch.cat((img_before, img_ref, img_after), 0)

        num_batch = images.shape[0] // 3
        '''
        if not self.training and self.transforms is not None:
            images = self.transforms(images)
        image_list = to_image_list(images)

        with torch.no_grad():
            predictions, features = self.maskrcnn(image_list)
        self.features, self.predictions = features, predictions
        
        return self.features

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 768)),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
