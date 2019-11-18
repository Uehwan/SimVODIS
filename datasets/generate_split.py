from path import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR', help='path to processed datasets')
parser.add_argument("--dump-root", type=str, default='splits', help="Where to create splits list")
parser.add_argument("--rgbd-step", type=int, default=10, help="step for each image to see moving")
parser.add_argument("--nyu-step", type=int, default=10, help="step for each image to see moving")
parser.add_argument("--scan-step", type=int, default=10, help="step for each image to see moving")
args = parser.parse_args()


class dataset(object):
    def __init__(self,
                 dataset_dir,
                 name,
                 img_exts,
                 step=1):
        self.dataset_dir = dataset_dir
        self.name = name
        self.path = self.dataset_dir / self.name
        self.img_exts = img_exts
        self.step = step
        self.get_frame_idx()
    
    def get_frame_idx(self):
        self.frame_idx = {}
        scenes = sorted(self.path.dirs())
        for scene in scenes:
            s = scene.basename()
            self.frame_idx[s] = []
            img_files = sorted(scene.files('*'+self.img_exts))
            for f in img_files[self.step:-self.step]:
                self.frame_idx[s].append(f.basename())
        
class KITTI_dataset(dataset):
    def __init__(self, *args, **kwargs):
        self.stereo = kwargs.pop('stereo')
        super(KITTI_dataset, self).__init__(*args, **kwargs)
        

    def get_frame_idx(self):
        self.frame_idx = {}
        scenes = sorted(self.path.dirs())
        for scene in scenes:
            s = scene.basename()
            if self.stereo:
                if s[-2:] == '03':
                    continue
            self.frame_idx[s] = []
            img_files = sorted(scene.files('*'+self.img_exts))
            for f in img_files[self.step:-self.step]:
                self.frame_idx[s].append(f.basename())

    

def main():
    args.dataset_dir = Path(args.dataset_dir)
    args.dump_root = Path(args.dump_root)
    kitti = KITTI_dataset(dataset_dir=args.dataset_dir, name='KITTI', img_exts='.jpg', stereo=True)
    malaga = dataset(dataset_dir=args.dataset_dir, name='Malaga', img_exts='.jpg')
    rgbd_slam = dataset(dataset_dir=args.dataset_dir, name='RGBD-SLAM', img_exts='.jpg', step=args.rgbd_step)
    nyu_depth = dataset(dataset_dir=args.dataset_dir, name='NYU_depth', img_exts='.jpg', step=args.nyu_step)
    scannet = dataset(dataset_dir=args.dataset_dir, name='Scannet', img_exts='.jpg', step=args.scan_step)
    
    args.dump_root.mkdir_p()
    datasets = [rgbd_slam, nyu_depth, malaga, scannet, kitti]
    train_files = []
    val_files = []
    np.random.seed(1234)
    for ds in datasets:
        for scene, frame_idx in ds.frame_idx.items():
            for idx in frame_idx:
                f_str = '{} {} {}'.format(ds.name / scene, idx[:-4], 'l')
                if np.random.random() < 0.1:
                    val_files.append(f_str)
                else:
                    train_files.append(f_str)

    np.random.shuffle(train_files)
    np.random.shuffle(val_files)
    with open(args.dump_root / 'train_files.txt', 'w') as tf:
        for line in train_files:
            tf.write(line + '\n')
    
    with open(args.dump_root / 'val_files.txt', 'w') as vf:
        for line in val_files:
            vf.write(line + '\n')

if __name__ == '__main__':
    main()