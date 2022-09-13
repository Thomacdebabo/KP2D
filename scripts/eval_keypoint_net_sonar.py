# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse

import torch
import os
import json

from datetime import datetime
from termcolor import colored
from torch.utils.data import DataLoader

from kp2dsonar.datasets.sonarsim import SonarSimLoader
from kp2dsonar.evaluation.evaluate import evaluate_keypoint_net_sonar
from kp2dsonar.networks.keypoint_net import KeypointNet
from kp2dsonar.networks.keypoint_resnet import KeypointResnet
from kp2dsonar.networks.ai84_keypointnet import ai84_keypointnet
from kp2dsonar.datasets.augmentations import resize_sample
from kp2dsonar.datasets.augmentations_sonar import to_tensor_sonar_sample
from kp2dsonar.datasets.noise_model import NoiseUtility
import glob
from kp2dsonar.datasets.augmentations import (ha_augment_sample, resize_sample,
                                              spatial_augment_sample,
                                              to_tensor_sample, normalize_sample, a8x_normalize_sample)
from kp2dsonar.utils.train_keypoint_net_utils import _set_seeds
def _print_result(result_dict):
    for k in result_dict.keys():
        print("%s: %.3f" %( k, result_dict[k]))
global _worker_init_fn
def _worker_init_fn(worker_id):
    """Worker init fn to fix the seed of the workers"""
    _set_seeds(42 + worker_id)
def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")
    parser.add_argument("--model_dir", required=False, type=str, help="Directory with models which will get evaluated", default='..\data\models\kp2dsonar')
    parser.add_argument("--device", required=False, type=str, help="cuda or cpu", default='cuda')
    parser.add_argument("--top_k", required=False, type=int, help="top-k value", default=300)
    parser.add_argument("--conf_threshold", required=False, type=float, help="Score threshold for keypoint detection", default=0.7)
    parser.add_argument("--debug", required=False, type=bool, help="toggle debug plots", default=False)
    parser.add_argument("--res", required=False, type=int, help="resolution in x direction", default=512)
    args = parser.parse_args()
    return args

def _load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model_args = checkpoint['config']['model']['params']

    print('Loaded KeypointNet from {}'.format(model_path))
    print('KeypointNet params {}'.format(model_args))
    print(checkpoint['config'])

    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        net_type = checkpoint['config']['model']['params']['keypoint_net_type']
    else:
        net_type = 'KeypointNet'  # default when no type is specified

    if net_type == 'KeypointNet':
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                   do_upsample=model_args['do_upsample'],
                                   do_cross=model_args['do_cross'], device = device)
    elif net_type == 'KeypointResnet':
        keypoint_net = KeypointResnet(device = device)
    elif net_type == 'KeypointMAX':
        keypoint_net = ai84_keypointnet( device = device)
    else:
        raise KeyError("net_type not recognized: " + str(net_type))

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.to(device)
    keypoint_net.eval()

    return keypoint_net, checkpoint['config']

def _get_eval_params(res, top_k):
    return [
        {'name': 'V6 V4_A4 config',
         'res': res,
         'top_k': top_k,
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
         'res': res,
         'top_k': top_k,
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
         'max_angle_div': 12},  # decided to not copy these values due to it being not very good at evaluating if big
        {'name': 'only row noise',
         'res': res,
         'top_k': top_k,
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
         'res': res,
         'top_k': top_k,
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
         'max_angle_div': 4},
        {'name': 'all the noise',
         'res': res,
         'top_k': top_k,
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
         'add_speckle_noise': True,
         'blur': True,
         'patch_ratio': 0.8,
         'scaling_amplitude': 0.2,
         'max_angle_div': 4}
    ]

class image_transforms():
    def __init__(self, noise_util):
        self.noise_util = noise_util
        self.transform = self._sonar_sim

    def _sonar_sim(self,sample):

        sample = resize_sample(sample, image_shape=self.noise_util.shape)

        sample = to_tensor_sample(sample)
        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = spatial_augment_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        sample = self.noise_util.squeeze(sample)
        sample = self.noise_util.sample_2_RGB(sample)
        # if self.noise_util.post_noise:
        #     sample = self.noise_util.add_noise_function(sample)

        #ample = normalize_sample(sample)
        return sample

    def _quantized_sonar(self, sample):
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)

        sample = to_tensor_sample(sample)
        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = spatial_augment_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        sample = self.noise_util.squeeze(sample)
        sample = self.noise_util.sample_2_RGB(sample)

        sample = a8x_normalize_sample(sample)
        return sample
    def __call__(self, sample):
        return self.transform(sample)

def main():

    args = parse_args()
    #Configuration - default: runs over all models found in ..\data\models\kp2dsonar
    torch.backends.cudnn.benchmark = True
    model_paths = glob.glob(os.path.join(args.model_dir,"*.ckpt"))
    top_k = args.top_k
    res = (args.res, args.res) #only square pics allowed at this point... Might cause problems with resnet if not square
    conf_threshold = args.conf_threshold
    debug = args.debug

    print("Running evaluation on:")
    print(model_paths)

    eval_params = _get_eval_params(res, top_k)
    evaluation_results = {}

    for model_path in model_paths:

        keypoint_net, config = _load_model(model_path, args.device)
        model_name = model_path.split('\\')[-1]

        results = []

        for params in eval_params:
            run_name = model_name + " - " + params['name']

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
            hp_dataset = SonarSimLoader(args.input_dir, noise_util,data_transform=data_transforms)
            data_loader = DataLoader(hp_dataset,
                                     batch_size=1,
                                     pin_memory=False,
                                     shuffle=False,
                                     num_workers=4,
                                     worker_init_fn=_worker_init_fn,
                                     sampler=None,
                                    drop_last=True)

            print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))

            result_dict = evaluate_keypoint_net_sonar(
                data_loader,
                keypoint_net,
                noise_util=noise_util,
                output_shape=params['res'],
                top_k=params['top_k'],
                conf_threshold=conf_threshold,
                use_color=True,
                device=args.device,
                debug=debug)

            results.append({'run_name': run_name,
                            'result': result_dict})

            _print_result(result_dict)

        evaluation_results[model_name] = {'model_config': config,
                                          'evaluation': results}

        evaluation_results['eval_params'] = eval_params

        #dt = datetime.now().strftime("_%d_%m_%Y__%H_%M_%S")
        pth = os.path.join('../data/eval', model_name + "_eval_result.json")

    with open(pth, "w") as f:
        json.dump(evaluation_results, f, indent=4, separators=(", ", ": "))
        print("Saved evaluation results to:",pth)

if __name__ == '__main__':
    main()

