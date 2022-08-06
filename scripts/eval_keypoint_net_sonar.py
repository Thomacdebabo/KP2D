# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import argparse
import os
import pickle
import random
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kp2d.datasets.sonarsim import SonarSimLoader
from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.evaluation.evaluate import evaluate_keypoint_net,evaluate_keypoint_net_sonar
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.datasets.augmentations import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample,to_tensor_sonar_sample)
from kp2d.datasets.noise_model import NoiseUtility


def image_transforms(noise_util):
    def train_transforms(sample):
        # sample = resize_sample(sample, image_shape=config.augmentation.image_shape)

        sample = noise_util.pol_2_cart_sample(sample)
        sample = noise_util.augment_sample(sample)

        sample = noise_util.filter_sample(sample)
        sample = noise_util.cart_2_pol_sample(sample)
        if noise_util.post_noise:
            sample = noise_util.add_noise_function(sample)
        sample = to_tensor_sonar_sample(sample)


        return sample

    return {'train': train_transforms}
def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

    noise_util = NoiseUtility((512,512),
                 fov=60,
                 r_min=0.1,
                 r_max=5.0,
                 super_resolution=1,
                 normalize=True,
                 preprocessing_gradient=True,
                 add_row_noise=True,
                 add_artifact=True,
                 add_sparkle_noise=False,
                 add_normal_noise= False,
                 add_speckle_noise=False,
                 blur=True,
                 patch_ratio=0.9,
                 scaling_amplitude=0.2)
    # Check model type
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        net_type = checkpoint['config']['model']['params']['keypoint_net_type']
    else:
        net_type = 'KeypointNet' # default when no type is specified

    # Create and load keypoint net
    if net_type == 'KeypointNet':
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'])
    else:
        keypoint_net = KeypointResnet()

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    eval_params = [{'res': (512, 512), 'top_k': 1500, }]

    for params in eval_params:
        data_transforms = image_transforms(noise_util)
        hp_dataset = SonarSimLoader(args.input_dir, noise_util,data_transform=data_transforms['train'])
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=0,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_sonar(
            data_loader,
            keypoint_net,
            noise_util=noise_util,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color=True)

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))

if __name__ == '__main__':
    main()

