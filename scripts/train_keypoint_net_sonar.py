# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import os
from datetime import datetime

import torch
import torch.optim as optim

from tqdm import tqdm

from kp2dsonar.evaluation.evaluate import evaluate_keypoint_net_sonar
from kp2dsonar.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2dsonar.utils.config import parse_train_file
from kp2dsonar.utils.logging import SummaryWriter, printcolor
from kp2dsonar.utils.train_keypoint_net_utils import (_set_seeds, sample_to_cuda,
                                      setup_datasets_and_dataloaders_sonar, setup_datasets_and_dataloaders_eval_sonar, image_transforms)
from train_keypoint_net import Trainer
from kp2dsonar.datasets.noise_model import NoiseUtility
#torch.autograd.set_detect_anomaly(True)

def _print_result(result_dict):
    for k in result_dict.keys():
        print("%s: %.3f" %( k, result_dict[k]))


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='KP2D training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args

def adjust_learning_rate(config, optimizer, epoch, decay=0.5, max_decays=4):
    """Sets the learning rate to the initial LR decayed by 0.5 every k epochs"""
    exponent = min(epoch // (config.model.scheduler.lr_epoch_divide_frequency / config.datasets.train.repeat), max_decays)
    decay_factor = (decay**exponent)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['original_lr'] * decay_factor
        printcolor('Changing {} network learning rate to {:8.6f}'.format(param_group['name'], param_group['lr']),
                   'red')

def model_submodule(model):
    """Get submodule of the model in the case of DataParallel, otherwise return
    the model itself. """
    return model.module if hasattr(model, 'module') else model

class TrainerSonar(Trainer):
    def __init__(self, config):

        self.noise_util = NoiseUtility(config.datasets.augmentation.image_shape,
                              fov=config.datasets.augmentation.fov,
                              r_min=config.datasets.augmentation.r_min,
                              r_max=config.datasets.augmentation.r_max,
                              patch_ratio=config.datasets.augmentation.patch_ratio,
                              scaling_amplitude=config.datasets.augmentation.scaling_amplitude,
                              max_angle_div=config.datasets.augmentation.max_angle_div,
                              super_resolution=config.datasets.augmentation.super_resolution,
                              amp=config.datasets.augmentation.amp,
                              artifact_amp=config.datasets.augmentation.artifact_amp,
                              preprocessing_gradient=config.datasets.augmentation.preprocessing_gradient,
                              add_row_noise=config.datasets.augmentation.add_row_noise,
                              add_normal_noise=config.datasets.augmentation.add_normal_noise,
                              add_artifact=config.datasets.augmentation.add_artifact,
                              add_sparkle_noise=config.datasets.augmentation.add_sparkle_noise,
                              blur=config.datasets.augmentation.blur,
                              add_speckle_noise=config.datasets.augmentation.add_speckle_noise,
                              normalize=config.datasets.augmentation.normalize,
                              device="cpu")
        super().__init__(config)
    def init_datasets(self, config):
        self.train_dataset, self.train_loader = setup_datasets_and_dataloaders_sonar(config.datasets, self.noise_util)
        self.hp_dataset, self.data_loader = setup_datasets_and_dataloaders_eval_sonar(config.datasets, self.noise_util)

    def evaluation(self, completed_epoch):
        # Set to eval mode
        self.model.eval()
        self.model.training = False

        use_color = self.config.model.params.use_color



        for params in self.eval_params:

            print('Loaded {} image pairs '.format(len(self.data_loader)))

            printcolor('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']))
            result_dict = evaluate_keypoint_net_sonar(self.data_loader,
                                                                model_submodule(self.model).keypoint_net,
                                                                output_shape=params['res'],
                                                                top_k=params['top_k'],
                                                                use_color=use_color)
            if self.summary:
                self.summary["evaluation"][completed_epoch] = result_dict
            _print_result(result_dict)

        # Save checkpoint
        if self.config.model.save_checkpoint:
            self.config['completed_epochs'] = completed_epoch
            current_model_path = os.path.join(self.config.model.checkpoint_path, 'model.ckpt')
            printcolor('\nSaving model (epoch:{}) at {}'.format(completed_epoch, current_model_path), 'green')
            torch.save(
            {
                'state_dict': model_submodule(model_submodule(self.model).keypoint_net).state_dict(),
                'config': self.config
            }, current_model_path)

            pth = os.path.join(self.config.model.checkpoint_path, "log.json")
            with open(pth, "w") as f:
                json.dump(self.summary, f, indent=4, separators=(", ", ": "))
                print("Saved evaluation results to:", pth)
            printcolor('Training complete, models saved in {}'.format(self.config.model.checkpoint_path), "green")


def main(file):
    """
    KP2D training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Parse config
    config = parse_train_file(file)
    print(config)
    print(config.arch)

    # Initialize horovod
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(n_threads)
    torch.backends.cudnn.benchmark = True

    noise_util = NoiseUtility(config.datasets.augmentation.image_shape,
                              fov=config.datasets.augmentation.fov,
                              r_min=config.datasets.augmentation.r_min,
                              r_max=config.datasets.augmentation.r_max,
                              patch_ratio=config.datasets.augmentation.patch_ratio,
                              scaling_amplitude=config.datasets.augmentation.scaling_amplitude,
                              max_angle_div=config.datasets.augmentation.max_angle_div,
                              super_resolution=config.datasets.augmentation.super_resolution,
                              amp=config.datasets.augmentation.amp,
                              artifact_amp=config.datasets.augmentation.artifact_amp,
                              preprocessing_gradient=config.datasets.augmentation.preprocessing_gradient,
                              add_row_noise=config.datasets.augmentation.add_row_noise,
                              add_normal_noise=config.datasets.augmentation.add_normal_noise,
                              add_artifact=config.datasets.augmentation.add_artifact,
                              add_sparkle_noise=config.datasets.augmentation.add_sparkle_noise,
                              blur=config.datasets.augmentation.blur,
                              add_speckle_noise=config.datasets.augmentation.add_speckle_noise,
                              normalize=config.datasets.augmentation.normalize,
                              device="cpu") #unfortunately noise model does not work with cuda due to cuda reinitialization issue
    printcolor('-'*25 + 'SINGLE GPU ' + '-'*25, 'cyan')

    if config.arch.seed is not None:
        _set_seeds(config.arch.seed)

    printcolor('-' * 25 + ' MODEL PARAMS ' + '-' * 25)
    printcolor(config.model.params, 'red')

    trainer = TrainerSonar(config)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args.file)
