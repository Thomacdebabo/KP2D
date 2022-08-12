# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import argparse

import torch
import os
import json

from datetime import datetime
from termcolor import colored
from torch.utils.data import DataLoader

from kp2d.datasets.sonarsim import SonarSimLoader
from kp2d.evaluation.evaluate import evaluate_keypoint_net,evaluate_keypoint_net_sonar
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.datasets.augmentations import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample,to_tensor_sonar_sample)
from kp2d.datasets.noise_model import NoiseUtility

def _load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model_args = checkpoint['config']['model']['params']



    print('Loaded KeypointNet from {}'.format(model_path))
    print('KeypointNet params {}'.format(model_args))
    print(checkpoint['config'])
    # Check model type
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        net_type = checkpoint['config']['model']['params']['keypoint_net_type']
    else:
        net_type = 'KeypointNet'  # default when no type is specified

    # Create and load keypoint net
    if net_type == 'KeypointNet':
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                   do_upsample=model_args['do_upsample'],
                                   do_cross=model_args['do_cross'])
    elif net_type == 'KeypointResnet':
        keypoint_net = KeypointResnet()
    else:
        raise KeyError("net_type not recognized: " + str(net_type))

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.to(device)
    keypoint_net.eval()

    return keypoint_net, checkpoint['config']

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
    #"C:\Users\Dr. Paul von Immel\Downloads\sonar_sim_noise\V_5.ckpt"
    #"D:\PycharmProjects\KP2D\data\models\kp2d\v4.ckpt"

    device = 'cuda'
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")
    parser.add_argument("--device", required=False, type=str, help="cuda or cpu")

    args = parser.parse_args()
    model_paths = [r"C:\Users\Dr. Paul von Immel\Downloads\sonar_sim_noise\V4_A4.ckpt",
                   r"C:\Users\Dr. Paul von Immel\Downloads\sonar_sim_noise\V_6.ckpt",
                   r"C:\Users\Dr. Paul von Immel\Downloads\sonar_sim_noise\V_5.ckpt",
                   r"C:\Users\Dr. Paul von Immel\Downloads\sonar_sim_noise\row.ckpt",
                   r"D:\PycharmProjects\KP2D\data\models\kp2d\v4.ckpt"]
    eval_params = [
                   {'name': 'V6 V4_A4 config',
                    'res': (512, 512),
                    'top_k': 1500,
                    'fov': 60,
                    'r_min': 0.1,
                    'r_max': 5.0,
                    'super_resolution': 1,
                    'normalize': True,
                    'preprocessing_gradient': True,
                    'add_row_noise': True,
                    'add_artifact': True,
                    'add_sparkle_noise': False,
                    'add_normal_noise': False,
                    'add_speckle_noise': False,
                    'blur': True,
                    'patch_ratio': 0.8,
                    'scaling_amplitude': 0.2,
                    'max_angle_div': 12},
                   {'name': 'V5 config',
                    'res': (512, 512),
                    'top_k': 1500,
                    'fov': 60,
                    'r_min': 0.1,
                    'r_max': 5.0,
                    'super_resolution': 1,
                    'normalize': True,
                    'preprocessing_gradient': True,
                    'add_row_noise': True,
                    'add_artifact': True,
                    'add_sparkle_noise': True,
                    'add_normal_noise': False,
                    'add_speckle_noise': False,
                    'blur': True,
                    'patch_ratio': 0.8,
                    'scaling_amplitude': 0.2,
                    'max_angle_div': 12}, #decided to not copy these values due to it being not very good at evaluating if big
                  {'name': 'only row noise',
                   'res': (512, 512),
                   'top_k': 1500,
                   'fov': 60,
                   'r_min': 0.1,
                   'r_max': 5.0,
                   'super_resolution': 1,
                   'normalize': False,
                   'preprocessing_gradient': False,
                   'add_row_noise': True,
                   'add_artifact': False,
                   'add_sparkle_noise': False,
                   'add_normal_noise': False,
                   'add_speckle_noise': False,
                   'blur': False,
                   'patch_ratio': 0.8,
                   'scaling_amplitude': 0.2,
                   'max_angle_div': 12},
                   {'name': 'no noise at all',
                    'res': (512, 512),
                    'top_k': 1500,
                    'fov': 60,
                    'r_min': 0.1,
                    'r_max': 5.0,
                    'super_resolution': 1,
                    'normalize': False,
                    'preprocessing_gradient': False,
                    'add_row_noise': False,
                    'add_artifact': False,
                    'add_sparkle_noise': False,
                    'add_normal_noise': False,
                    'add_speckle_noise': False,
                    'blur': False,
                    'patch_ratio': 0.8,
                    'scaling_amplitude': 0.2,
                    'max_angle_div': 4}
                   ]
    for model_path in model_paths:
        keypoint_net, config = _load_model(model_path, device)
        print(config)

        results = []
        for params in eval_params:
            noise_util = NoiseUtility(params['res'],
                                      fov=params['fov'],
                                      r_min=params['r_min'],
                                      r_max=params['r_max'],
                                      super_resolution=params['super_resolution'],
                                      normalize=params['normalize'],
                                      preprocessing_gradient=params['preprocessing_gradient'],
                                      add_row_noise=params['add_row_noise'],
                                      add_artifact=params['add_artifact'],
                                      add_sparkle_noise=params['add_sparkle_noise'],
                                      add_normal_noise=params['add_normal_noise'],
                                      add_speckle_noise=params['add_speckle_noise'],
                                      blur=params['blur'],
                                      patch_ratio=params['patch_ratio'],
                                      scaling_amplitude=params['scaling_amplitude'],
                                      max_angle_div=params['max_angle_div'])


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
            rep, loc, c1, c5, c10, mscore, up, md = evaluate_keypoint_net_sonar(
                data_loader,
                keypoint_net,
                noise_util=noise_util,
                output_shape=params['res'],
                top_k=params['top_k'],
                use_color=True, device=device)
            results.append({'params': params,
                            'model_config': config,
                            'eval_result':
                                { 'Repeatability':rep.item(),
                                'Localization Error':loc.item(),
                                'Correctness d1':c1.item(),
                                'Correctness d5':c5.item(),
                                'Correctness d10':c10.item(),
                                'MScore':mscore.item(),
                                'Useful points ratio ':up.item(),
                                'Mean distance (debug)':md.item()}})

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d5 {:.3f}'.format(c5))
            print('Correctness d10 {:.3f}'.format(c10))
            print('MScore {:.3f}'.format(mscore))
            print('Useful points ratio  {:.3f}'.format(up))
            print('Mean distance (debug) {:.3f}'.format(md))

        dt = datetime.now().strftime("_%d_%m_%Y__%H_%M_%S")
        pth = os.path.join('../data/eval', dt + "_eval_result.json")
        with open(pth, "w") as f:
            json.dump(results, f, indent=4, separators=(", ", ": "))
            print("Saved evaluation results to:",pth)

if __name__ == '__main__':
    main()

