# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""Default kp2d configuration parameters (overridable in configs/*.yaml)
"""

import os
from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''       # Run name
cfg.debug = False   # Debugging flag
cfg.device = 'cuda'
########################################################################################################################
### ARCH
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 42                # Random seed for Pytorch/Numpy initialization
cfg.arch.epochs = 50                # Maximum number of epochs
########################################################################################################################
### WANDB
########################################################################################################################
cfg.wandb = CN()
cfg.wandb.dry_run = True                                 # Wandb dry-run (not logging)
cfg.wandb.name = ''                                      # Wandb run name
cfg.wandb.project = os.environ.get("WANDB_PROJECT", "")  # Wandb project
cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")    # Wandb entity
cfg.wandb.tags = []                                      # Wandb tags
cfg.wandb.dir = ''                                       # Wandb save folder
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.checkpoint_path = '/data/experiments/kp2d/'              # Checkpoint path for model saving
cfg.model.save_checkpoint = True

########################################################################################################################
### MODEL.SCHEDULER
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.decay = 0.5                                # Scheduler decay rate
cfg.model.scheduler.lr_epoch_divide_frequency = 40             # Schedule number of epochs when to decay the initial learning rate by decay rate
########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.learning_rate = 0.001
cfg.model.optimizer.weight_decay = 0.0
########################################################################################################################
### MODEL.PARAMS
########################################################################################################################
cfg.model.params = CN()
cfg.model.params.debug = False
cfg.model.params.device = 'cuda'
cfg.model.params.keypoint_loss_weight = 1.0                 # Keypoint loss weight
cfg.model.params.descriptor_loss_weight = 1.0               # Descriptor loss weight
cfg.model.params.score_loss_weight = 1.0                    # Score loss weight
cfg.model.params.use_color = True                           # Use color or grayscale images
cfg.model.params.with_io = True                             # Use IONet
cfg.model.params.do_upsample = True                         # Upsample descriptors
cfg.model.params.do_cross = True                            # Use cross-border keypoints
cfg.model.params.descriptor_loss = True                     # Use hardest negative mining descriptor loss
cfg.model.params.keypoint_net_type = 'KeypointNet'          # Type of keypoint network. Supported ['KeypointNet', 'KeypointResnet']
########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CN()
########################################################################################################################
### DATASETS.AUGMENTATION
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = (512, 512)              # Image shape
cfg.datasets.augmentation.jittering = (0, 0, 0, 0)     # Color jittering values
cfg.datasets.augmentation.fov = 60
cfg.datasets.augmentation.r_min = 0.1
cfg.datasets.augmentation.r_max = 5.0
cfg.datasets.augmentation.patch_ratio = 0.95
cfg.datasets.augmentation.scaling_amplitude = 0.1
cfg.datasets.augmentation.max_angle_div = 18
cfg.datasets.augmentation.super_resolution = 1
cfg.datasets.augmentation.amp = 70
cfg.datasets.augmentation.artifact_amp = 200
cfg.datasets.augmentation.mode = 'sonar_sim'
cfg.datasets.augmentation.preprocessing_gradient = True
cfg.datasets.augmentation.add_row_noise = True
cfg.datasets.augmentation.add_normal_noise = False
cfg.datasets.augmentation.add_artifact = True
cfg.datasets.augmentation.add_sparkle_noise = False
cfg.datasets.augmentation.blur = True
cfg.datasets.augmentation.add_speckle_noise = False
cfg.datasets.augmentation.normalize = True

########################################################################################################################
### DATASETS.TRAIN
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 2                                   # Training batch size
cfg.datasets.train.num_workers = 0   #16 for Euler                                  # Training number of workers
cfg.datasets.train.path = '/data/datasets/kp2d/coco/train2017/'        # Training data path (COCO dataset)
cfg.datasets.train.repeat = 1                                          # Number of times training dataset is repeated per epoch
########################################################################################################################
### DATASETS.VAL
########################################################################################################################
cfg.datasets.val = CN()
cfg.datasets.val.path = '/data/datasets/kp2d/HPatches/'     # Validation data path (HPatches)
########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''                 # Run configuration file
cfg.default = ''                # Run default configuration file
cfg.wandb.url = ''              # Wandb URL
########################################################################################################################

def get_cfg_defaults():
    return cfg.clone()
