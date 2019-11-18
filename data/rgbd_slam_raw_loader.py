from __future__ import division
import argparse
import numpy as np
from path import Path
from pebble import ProcessPool
import scipy.misc
import sys
from tqdm import tqdm
from collections import Counter
import torch

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--dump-root", type=str, default='dump', help="Where to dump the data")
parser.add_argument("--with-depth", action='store_true',
                    help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
parser.add_argument("--with-pose", action='store_true',
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--height", type=int, default=192, help="image height")
parser.add_argument("--width", type=int, default=640, help="image width")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz, rw -- [B, 7]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches


class RGBDSlamRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=192,
                 img_width=640,
                 depth_scale=5000,
                 get_depth=False,
                 get_pose=False):
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.depth_scale = depth_scale
        self.get_depth = get_depth
        self.get_pose = get_pose
        self.call_intrinsics()
        self.collect_train_folders()
    
    def call_intrinsics(self):
        self.intrinsics = {}
        freiburg1 = np.array([[517.3,     0, 318.6,     0],
                              [    0, 516.5, 255.3,     0],
                              [    0,     0,     1,     0],
                              [    0,     0,     0,     1]], dtype=np.float32)
        freiburg2 = np.array([[520.9,     0, 325.1,     0],
                              [    0, 521.0, 249.7,     0],
                              [    0,     0,     1,     0],
                              [    0,     0,     0,     1]], dtype=np.float32)
        freiburg3 = np.array([[535.4,     0, 320.1,     0],
                              [    0, 539.2, 247.6,     0],
                              [    0,     0,     1,     0],
                              [    0,     0,     0,     1]], dtype=np.float32)
        self.intrinsics['freiburg1'] = freiburg1
        self.intrinsics['freiburg2'] = freiburg2
        self.intrinsics['freiburg3'] = freiburg3

    def collect_train_folders(self):
        self.scenes = []
        drive_set = sorted(self.dataset_dir.dirs())
        for dr in drive_set:
            self.scenes.append(dr)

    def associate_all(self, scene_data, rgb, depth, gt):
        rgb_depth_match = associate(rgb, depth)
        rgb_gt_match = associate(rgb, gt)
        match = {}
        matched_pose = []
        
        for i, j in rgb_depth_match:
            match[i] = j
        
        for i, j in rgb_gt_match:
            if i in match:
                scene_data['rgb_frame_id'].append("{:10.6f}".format(i))
                scene_data['depth_frame_id'].append("{:10.6f}".format(match[i]))
                matched_pose.append(j)
        
        return matched_pose

    def get_intrinsics(self, cid, zoom_x, zoom_y):
        intrinsics = self.intrinsics[cid]
        intrinsics[0] *= zoom_x / self.img_width
        intrinsics[1] *= zoom_y / self.img_height
        return intrinsics

    def collect_scene_data(self, drive):
        drive_info = drive.name.split('_')
        cid = drive_info[2]
        scene_data = {'cid':cid, 'dir':drive, 'rgb_frame_id':[], 'depth_frame_id':[], 'pose':[], 'rel_path':drive.name}
        pose_gt = np.genfromtxt(drive/'groundtruth.txt') # format timestamp, translation vector, quaternion
        
        rgb_list = read_file_list(drive/'rgb.txt')
        depth_list = read_file_list(drive/'depth.txt')
        gt_list = read_file_list(drive/'groundtruth.txt')
        # Compare timestamp of rgb image and depth image and match most similar timestamps
        sync_rgb_depth = associate(rgb_list, depth_list)
        # Compare timestamp of rgb image and pose groundtruth and match most similar timestamps
        sync_rgb_pose = associate(rgb_list, gt_list)

        # From matched timestamp, compare three types of data's timestamp and append if matched
        matches = self.associate_all(scene_data, rgb_list, depth_list, gt_list)

        for pose in pose_gt:
                if pose[0] in matches:
                    pose_vec = torch.from_numpy(pose[1:]).view(1, -1)
                    pose_mat = pose_vec2mat(pose_vec)
                    scene_data['pose'].append(pose_mat.numpy())
        
        assert len(scene_data['rgb_frame_id']) == len(scene_data['depth_frame_id'])
        assert len(scene_data['pose']) == len(scene_data['rgb_frame_id'])
        sample = self.load_image(scene_data, 0)
        if sample is None:
            return []
        scene_data['intrinsics'] = self.get_intrinsics(cid, sample[1], sample[2])
        return scene_data
    
    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i):
            sample = {'img': self.load_image(scene_data, i)[0], 'rgb_id':scene_data['rgb_frame_id'][i]}
            if self.get_depth:
                sample['depth'] = self.load_depth(scene_data, i)[0]
                sample['depth_id'] = scene_data['depth_frame_id'][i]
            if self.get_pose:
                sample['pose'] = scene_data['pose'][i]
            return sample
        
        for (i, frame_id) in enumerate(scene_data['rgb_frame_id']):
            yield construct_sample(scene_data, i)

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'rgb/{}.png'.format(scene_data['rgb_frame_id'][tgt_idx])

        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        img = self.crop_image(img)
        img = img / self.depth_scale
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
        if zoom_x != 1 and zoom_y != 1:
            # print("img resize")
            img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y
    
    def load_depth(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'depth/{}.png'.format(scene_data['depth_frame_id'][tgt_idx])
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        img = self.crop_image(img)
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
        if zoom_x != 1 and zoom_y != 1:
            # print("img resize")
            img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def crop_image(self, image):
        h, w = image.shape[0], image.shape[1]
        bbox_h = [h//2 - self.img_height//2, h//2 + self.img_height//2]
        bbox_w = [w//2 - self.img_width//2, w//2 + self.img_width//2]
        image = image[bbox_h[0]:bbox_h[1], bbox_w[0]:bbox_w[1]]
        # print(image.shape)
        return image


def dump_example(args, scene):
    scene_data = data_loader.collect_scene_data(scene)
    assert len(scene_data) != 0
    dump_dir = args.dump_root/scene_data['rel_path']

    dump_dir.makedirs_p()
    intrinsics = scene_data['intrinsics']

    dump_cam_file = dump_dir/'cam.txt'

    np.savetxt(dump_cam_file, intrinsics)
    poses_file = dump_dir/'poses.txt'
    poses = []

    idx = 0
    for sample in data_loader.get_scene_imgs(scene_data):
        img = sample["img"]
        dump_img_file = dump_dir/'{:010d}.jpg'.format(idx)
        scipy.misc.imsave(dump_img_file, img)
        if "pose" in sample.keys():
            poses.append(sample["pose"].tolist())
        if "depth" in sample.keys():
            dump_depth_file = dump_dir/'{:010d}.npy'.format(idx)
            np.save(dump_depth_file, sample["depth"])
        idx += 1
    if len(poses) != 0:
        np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

    if len(dump_dir.files('*.jpg')) < 3:
        dump_dir.rmtree()

def main():
    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader
    data_loader = RGBDSlamRawLoader(args.dataset_dir,
                                    img_height=args.height,
                                    img_width=args.width,
                                    get_depth=args.with_depth,
                                    get_pose=args.with_pose)

    n_scenes = len(data_loader.scenes)
    print('Found {} potential scenes'.format(n_scenes))
    print('Retrieving frames')
    if args.num_threads == 1:
        for scene in tqdm(data_loader.scenes):
            dump_example(args, scene)
    else:
        with ProcessPool(max_workers=args.num_threads) as pool:
            tasks = pool.map(dump_example, [args]*n_scenes, data_loader.scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e
    '''
    print('Generating train val lists')
    np.random.seed(8964)
    # to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
    subdirs = args.dump_root.dirs()
    canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    with open(args.dump_root / 'train.txt', 'w') as tf:
        with open(args.dump_root / 'val.txt', 'w') as vf:
            for pr in tqdm(canonic_prefixes):
                corresponding_dirs = args.dump_root.dirs('{}*'.format(pr))
                if np.random.random() < 0.1:
                    for s in corresponding_dirs:
                        vf.write('{}\n'.format(s.name))
                else:
                    for s in corresponding_dirs:
                        tf.write('{}\n'.format(s.name))
    '''

if __name__ == '__main__':
    main()

