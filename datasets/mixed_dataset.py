# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import PIL.Image as pil
from path import Path

from .mono_dataset import MonoDataset


class MixedDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(MixedDataset, self).__init__(*args, **kwargs)
        self.full_res_shape = (640, 192)
    
    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        data_path = Path(self.data_path)
        self.K = np.genfromtxt(data_path / folder / 'cam.txt', dtype=np.float32)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = Path("{:010d}{}".format(frame_index, self.img_ext))
        data_path = Path(self.data_path)
        image_path = data_path / folder / f_str
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        pass
