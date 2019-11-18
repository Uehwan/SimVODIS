from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import scipy.io

from layers import disp_to_depth
from options import MonodepthOptions
import networks

from maskrcnn_benchmark.config import cfg


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    height, width = 192, 640

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    data_images = sorted(os.listdir(os.path.join(opt.data_path, 'image')))

    config_file = "./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(config_file)
    cfg.freeze()

    normalize_transform = transforms.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    to_bgr_transform = transforms.Lambda(lambda x: x * 255)
    transform_simvodis = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize((height * 2, width * 2)),
            transforms.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )

    maskrcnn_path = "./e2e_mask_rcnn_R_50_FPN_1x.pth"
    encoder = networks.ResnetEncoder(cfg, maskrcnn_path)
    depth_decoder = networks.DepthDecoder(scales=opt.scales)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps, depths_gt = [], []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    if 'RGBD' in opt.data_path:
        pairing = open(os.path.join(opt.data_path, 'association.txt')).readlines()
        pairing = [item.split()[1:4:2] for item in pairing]
        pairing = {item[0][4:]: item[1][6:] for item in pairing}

    with torch.no_grad():
        for one_image in data_images:
            file_path_img = os.path.join(opt.data_path, 'image', one_image)
            img_mat = pil_loader(file_path_img)
            if '7Scenes' in file_path_img:
                img_mat = img_mat.crop((0, int((480 - 192)/2), 640, int((480 + 192)/2)))
                depth_mat = cv2.imread(os.path.join(opt.data_path, 'depth', one_image.split('.')[0] + '.depth.png'),
                                       cv2.IMREAD_ANYDEPTH)
                depth_mat = depth_mat[int((480-192)/2):int((480+192)/2), :]
                mask = (depth_mat != 65535)
                depth_mat = (depth_mat * mask) / 1000
                gt_depth = np.expand_dims(depth_mat, axis=0)
                depths_gt.append(gt_depth)
            elif 'Make3D' in file_path_img:
                img_mat = img_mat.crop((0, int((2272 - 511)/2), 1704, int((2272 + 511)/2)))
                mat = scipy.io.loadmat(os.path.join(opt.data_path, "depth", "depth_sph_corr-{}.mat".format(one_image[4:-4])))
                ratio = 4.4
                depth_new_height = 55 / ratio
                gt_depth = mat["Position3DGrid"][:, :, 3][int((55 - depth_new_height) / 2):int((55 + depth_new_height) / 2)]
                gt_depth = np.expand_dims(gt_depth, axis=0)
                depths_gt.append(gt_depth)
            elif 'RGBD' in file_path_img:
                img_mat = img_mat.crop((0, int((480 - 192) / 2), 640, int((480 + 192) / 2)))
                depth_mat = cv2.imread(os.path.join(opt.data_path, 'depth', pairing[one_image]),
                                       cv2.IMREAD_ANYDEPTH)
                depth_mat = depth_mat[int((480 - 192) / 2):int((480 + 192) / 2), :]
                mask = (depth_mat != 65535)
                depth_mat = (depth_mat * mask) / 1000
                gt_depth = np.expand_dims(depth_mat, axis=0)
                depths_gt.append(gt_depth)
            input_color = transform_simvodis(img_mat).cuda()
            input_color = input_color.unsqueeze(0)

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    gt_depths = np.concatenate(depths_gt)

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
