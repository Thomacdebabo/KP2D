# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
from kp2dsonar.evaluation.evaluate import evaluate_keypoint_net_sonar
from kp2dsonar.utils.config import parse_train_file
from kp2dsonar.utils.logging import printcolor
from kp2dsonar.utils.train_keypoint_net_utils import (_set_seeds, setup_datasets_and_dataloaders_sonar, setup_datasets_and_dataloaders_eval_sonar)
from train_keypoint_net import Trainer, parse_args, model_submodule
from kp2dsonar.datasets.noise_model import NoiseUtility

def _print_result(result_dict):
    for k in result_dict.keys():
        print("%s: %.3f" %( k, result_dict[k]))


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

    def _evaluate(self, completed_epoch, params):
        use_color = self.config.model.params.use_color
        result_dict = evaluate_keypoint_net_sonar(self.data_loader,
                                                  model_submodule(self.model).keypoint_net,
                                                  output_shape=params['res'],
                                                  top_k=params['top_k'],
                                                  use_color=use_color)
        if self.summary:
            self.summary["evaluation"][completed_epoch] = result_dict
        _print_result(result_dict)

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

    if config.arch.seed is not None:
        _set_seeds(config.arch.seed)

    printcolor('-' * 25 + ' MODEL PARAMS ' + '-' * 25)
    printcolor(config.model.params, 'red')

    trainer = TrainerSonar(config)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args.file)
