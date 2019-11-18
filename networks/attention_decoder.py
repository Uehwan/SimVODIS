# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class AttentionDecoder(nn.Module):
    def __init__(self, scales=range(1), num_output_channels=3):
        super(AttentionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        
        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )
        
        self.attention_pred = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        if 1 in self.scales:
            self.attention_pred1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, num_output_channels, kernel_size=3, stride=1, padding=0),
                nn.Sigmoid()
            )
        if 2 in self.scales:
            self.attention_pred2 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, num_output_channels, kernel_size=3, stride=1, padding=0),
                nn.Sigmoid()
            )
        if 3 in self.scales:
            self.attention_pred3 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, num_output_channels, kernel_size=3, stride=1, padding=0),
                nn.Sigmoid()
            )
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature_maps):
        self.outputs = {}

        feats = list(reversed(feature_maps))

        x = self.deconv1(self.conv1(feats[0]))
        
        x = self.deconv2(torch.cat([self.conv2(feats[1]), x], dim=1))
        if 3 in self.scales:
            self.outputs[("attention", 3)] = self.attention_pred3(x)
        
        x = self.deconv3(torch.cat([self.conv3(feats[2]), x], dim=1))
        if 2 in self.scales:
            self.outputs[("attention", 2)] = self.attention_pred2(x)
        
        x = self.deconv4(torch.cat([self.conv4(feats[3]), x], dim=1))
        if 1 in self.scales:
            self.outputs[("attention", 1)] = self.attention_pred1(x)
        
        x = self.deconv5(torch.cat([self.conv5(feats[4]), x], dim=1))
        x = self.attention_pred(x)
        self.outputs[("attention", 0)] = x
        return self.outputs
