# SimVODIS
[Simultaneous Visual Odometry, Object Detection, and Instance Segmentation.](https://arxiv.org/abs/1911.05939)
SimVODIS extracts both semantic and physical attributes from a sequence of image frames. SimVODIS evaluates the relative pose between frames, while detecting objects and segementing the object boundaries. During the process, depth can be optionally estimated.

## Notification
Thank you for visiting our repo. :)
We are reformatting the project for distribution.
We expect we could finish the reformatting in a few weeks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

* Ubuntu 16.04+
* CUDA >= 9.0
* Python 3.6+
* [Pytorch 1.0.0 from a nightly release](https://pytorch.org/get-started/previous-versions/)
* [MaskRCNN (included in this project)](https://github.com/facebookresearch/maskrcnn-benchmark)
* GCC >= 4.9

### Installation

We tested the code in the following environments: 1) CUDA 9.0 on Ubuntu 16.04 and 2) CUDA 10.1 on Ubuntu 18.04. SimVODIS may work in other environments, but you might need to modify a part of the code. We recommend you using Anaconda for the environment setup.

```bash
conda create --name SimVODIS python=3.6.7
conda activate SimVODIS
conda install ipython
pip install ninja yacs cython matplotlib tqdm opencv-python
# conda install -c pytorch pytorch-nightly=1.0 torchvision=0.2.2 cudatoolkit=10.0
conda install -c pytorch pytorch-nightly=1.0 torchvision cudatoolkit=9.0

# install SimVODIS
git clone https://github.com/Uehwan/SimVODIS.git
cd SimVODIS
# the following will install the lib with symbolic links,
# so that you can modify the files if you want and won't need to re-build it
python setup.py build develop

pip install tensorboardX
conda install -c anaconda path.py scipy=1.2
```

### Pretrained Mask-RCNN model

Download one of the following pretrained Mask-RCNN models and place it under the root directory
- [R-50-FPN](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth)
- [R-101-FPN](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x.pth)
- [X-101-32x8d-FPN](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth)

For more detailed information on the Mask-RCNN models, refer to the [Facebook Mask-RCNN benchmark repo](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/MODEL_ZOO.md)


## Data Preparation

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website of KITTI. Placing the dataset on SSD would increase the training speed.

## Training
```bash
python train.py
```

## Evaluation
```bash
python test_depth.py
```

```bash
python test_pose.py
```

## Performance

### Pretrained Networks

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citations

Please consider citing this project in your publications if you find this helpful.
The following is the BibTeX.

```
@article{kim2019simvodis,
  title={SimVODIS: Simultaneous Visual Odometry, Object Detection, and Instance Segmentation},
  author={Ue-Hwan Kim, Se-Ho Kim and Jong-Hwan Kim},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence, submitted},
  year={2019}
}
```

## Acknowledgments

We base our project on the following repositories
* [Monodepth2](https://github.com/nianticlabs/monodepth2)
* [MaskRCNN](https://github.com/facebookresearch/maskrcnn-benchmark)

