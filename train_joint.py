# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import torch.optim as optim
import time
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_optimizer_simvodis
from maskrcnn_benchmark.engine.trainer import do_train_one_step
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

from trainer import Trainer

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


class JointTrainer(Trainer):
    def __init__(self, options):
        super().__init__(options, joint_training=True)
        
        self.optimizer_maskrcnn = make_optimizer_simvodis(self.cfg, self.models["encoder"].maskrcnn, self.opt.learning_rate_maskrcnn)
        self.scheduler_maskrcnn = optim.lr_scheduler.StepLR(
            self.optimizer_maskrcnn, self.opt.scheduler_step_size, self.opt.scheduler_gamma
        ) # make_lr_scheduler(self.cfg, self.optimizer_maskrcnn)
        
        # Initialize mixed-precision training
        # self.models["encoder"].maskrcnn, self.optimizer_maskrcnn = amp.initialize(
        #     self.models["encoder"].maskrcnn, self.optimizer_maskrcnn, 'O0')

        self.arguments = {}
        self.arguments["iteration"] = 0
        self.data_loader_maskrcnn = make_data_loader(
            self.cfg,
            is_train=True,
            is_distributed=False,
            start_iter=self.arguments["iteration"],
        )
    
    def run_one_step_simvodis(self):
        """Run a single training step
        """
        self.model_lr_scheduler.step()
        self.set_train()

        inputs = next(iter(self.train_loader))

        before_op_time = time.time()

        outputs, losses = self.process_batch(inputs)

        self.model_optimizer.zero_grad()
        losses["loss"].backward()
        self.model_optimizer.step()

        duration = time.time() - before_op_time

        if self.arguments["iteration"] % (self.opt.save_frequency // 2) == 0:
            self.log_time(self.arguments["iteration"], duration, losses["loss"].cpu().data)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("train", inputs, outputs, losses)
            self.val()

        self.step += 1

    def train_joint(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        print("Training iteratively for the whole SimVODIS and Mask-RCNN")
        
        for _ in range(self.opt.num_epochs):
            do_train_one_step(
                self.cfg,
                self.models['encoder'].maskrcnn,
                self.data_loader_maskrcnn,
                self.optimizer_maskrcnn,
                self.scheduler_maskrcnn,
                self.device,
                self.arguments,
                self.opt
            )
            self.run_one_step_simvodis()
            if self.arguments["iteration"] % self.opt.save_frequency == 0:
                self.save_model()
                self.epoch += 1


if __name__ == "__main__":
    from options import MonodepthOptions

    options = MonodepthOptions(withMaskRCNN=True)
    args = options.parse()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    trainer = JointTrainer(args)
    trainer.train_joint()
