# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.ReLU()
    )


class PoseDecoder(nn.Module):
    def __init__(self, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.vo_conv1 = conv(256 * 3, 256, kernel_size=7)
        self.vo_conv2 = conv(256, 128, kernel_size=5)
        self.vo_conv3 = conv(128, 64, kernel_size=5)
        self.vo_conv4 = nn.Conv2d(64, 6 * num_frames_to_predict_for, 3)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, input_features):
        first_features = [f[0] for f in input_features]
        cat_features = torch.cat(first_features, 1)

        out = cat_features
        out = self.vo_conv4(
            self.vo_conv3(
                self.vo_conv2(
                    self.vo_conv1(out)
                )
            )
        )
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
