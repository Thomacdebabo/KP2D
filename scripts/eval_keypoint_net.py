# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2dsonar/v4.pth --input_dir /data/datasets/kp2dsonar/HPatches/

import argparse

import torch
from termcolor import colored
from torch.utils.data import DataLoader

from kp2dsonar.datasets.patches_dataset import PatchesDataset
from kp2dsonar.evaluation.evaluate import evaluate_keypoint_net
from kp2dsonar.networks.keypoint_net import KeypointNet
from kp2dsonar.networks.keypoint_resnet import KeypointResnet
from kp2dsonar.networks.ai84_keypointnet import ai84_keypointnet

def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']
    mode = 'default'
    # Create and load keypoint net
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        net_type = checkpoint['config']['model']['params']['keypoint_net_type']
    else:
        net_type = 'KeypointNet' # default when no type is specified

    # Create and load keypoint net
    if net_type == 'KeypointNet':
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'], device="cuda")
    elif net_type == 'KeypointResnet':
        keypoint_net = KeypointResnet(device="cuda")

    elif net_type == 'KeypointMAX':
        keypoint_net = ai84_keypointnet( device = "cuda")
        mode = 'quantized_default'
    else:
        raise KeyError ("net_type not recognized: " + str(net_type))
    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))
    print(checkpoint['config'])

    eval_params = [{'res': (320, 240), 'top_k': 300, }] if net_type is KeypointNet else [{'res': (320, 256), 'top_k': 300, }] # KeypointResnet needs (320,256)
    eval_params += [{'res': (320, 240), 'top_k': 1500, }]
    eval_params += [{'res': (640, 480), 'top_k': 1000, }]

    for params in eval_params:
        hp_dataset = PatchesDataset(root_dir=args.input_dir, use_color=True,
                                    output_shape=params['res'], type='a', mode=mode)
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
            data_loader,
            keypoint_net,
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
