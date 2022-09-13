# Copyright 2020 Toyota Research Institute.  All rights reserved.

from kp2dsonar.models.KeypointNetwithIOLossSonar import KeypointNetWithIOLossSonar
from math import pi
from torch.utils.data import ConcatDataset, DataLoader

from kp2dsonar.datasets.augmentations import (ha_augment_sample, resize_sample,
                                              spatial_augment_sample,
                                              to_tensor_sample, normalize_sample, a8x_normalize_sample)
from kp2dsonar.datasets.augmentations_sonar import to_tensor_sonar_sample

from kp2dsonar.datasets.sonarsim import SonarSimLoader
from kp2dsonar.datasets.noise_model import NoiseUtility
from kp2dsonar.evaluation.evaluate import evaluate_keypoint_net_sonar

from kp2dsonar.utils.train_keypoint_net_utils import (_set_seeds, Trainer, model_submodule)
def _print_result(result_dict):
    for k in result_dict.keys():
        print("%s: %.3f" %( k, result_dict[k]))


global _worker_init_fn
def _worker_init_fn(worker_id):
    """Worker init fn to fix the seed of the workers"""
    _set_seeds(42 + worker_id)

class image_transforms():
    def __init__(self, noise_util,config):
        self.noise_util = noise_util
        self.config = config
        mode = config.augmentation.mode
        self.transform = getattr(self,  "_" + mode)

    def _sonar_sim(self,sample):

        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)

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

        sample = normalize_sample(sample)

        return sample

    def _sonar_real(self, sample): #TODO

        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        sample = to_tensor_sonar_sample(sample)
        sample = normalize_sample(sample)

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

    def _default(self, sample):

        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=self.config.augmentation.jittering,
                                   patch_ratio=self.config.augmentation.patch_ratio,
                                   scaling_amplitude=self.config.augmentation.scaling_amplitude,
                                   max_angle=pi/self.config.augmentation.max_angle_div)
        sample = normalize_sample(sample)

        return sample

    def __call__(self, sample):
        return self.transform(sample)

class image_transforms_eval():
    def __init__(self, noise_util,config):
        self.noise_util = noise_util
        self.config = config
        mode = config.augmentation.mode
        self.transform = getattr(self,  "_" + mode)

    def _sonar_sim(self,sample):
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)

        sample = to_tensor_sample(sample)
        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        sample = self.noise_util.squeeze(sample)
        sample = self.noise_util.sample_2_RGB(sample)

        sample = normalize_sample(sample)

        return sample

    def _sonar_real(self, sample): #TODO
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = to_tensor_sample(sample)
        sample = normalize_sample(sample)

        return sample

    def _quantized_sonar(self, sample): #TODO: implement
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)

        sample = to_tensor_sample(sample)
        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        sample = self.noise_util.squeeze(sample)
        sample = self.noise_util.sample_2_RGB(sample)

        sample = a8x_normalize_sample(sample)
        return sample
    def _default(self, sample):
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = to_tensor_sample(sample)
        sample = normalize_sample(sample)
        return sample

    def __call__(self, sample):
        return self.transform(sample)

def setup_datasets_and_dataloaders_sonar(config,noise_util):
    """Prepare datasets for training, validation and test."""


    data_transforms = image_transforms(noise_util,config)
    train_dataset = SonarSimLoader(config.train.path, noise_util,data_transform=data_transforms)
    # Concatenate dataset to produce a larger one
    if config.train.repeat > 1:
        train_dataset = ConcatDataset([train_dataset for _ in range(config.train.repeat)])

    # Create loaders

    sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              pin_memory=False, # pin memory on seems to create an error
                              shuffle=True,
                              num_workers=config.train.num_workers,
                              worker_init_fn=_worker_init_fn,
                              sampler=sampler,
                              drop_last=True)

    return train_dataset, train_loader

def setup_datasets_and_dataloaders_eval_sonar(config,noise_util):
    """Prepare datasets for training, validation and test."""

    data_transforms = image_transforms_eval(noise_util,config)
    train_dataset = SonarSimLoader(config.val.path, noise_util,data_transform=data_transforms)
    # Concatenate dataset to produce a larger one
    if config.train.repeat > 1:
        train_dataset = ConcatDataset([train_dataset for _ in range(config.train.repeat)])

    # Create loaders

    sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              pin_memory=False, # pin memory on seems to create an error
                              shuffle=True,
                              num_workers=config.train.num_workers,
                              worker_init_fn=_worker_init_fn,
                              sampler=sampler,
                              drop_last=True)
    return train_dataset, train_loader

class TrainerSonar(Trainer):
    def __init__(self, config):
        self.debug = True
        self.conf_threshold = 0.7
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
    def init_model(self,config):
        self.model = KeypointNetWithIOLossSonar(self.noise_util, **config.model.params)
    def _evaluate(self, completed_epoch, params):
        use_color = self.config.model.params.use_color
        result_dict = evaluate_keypoint_net_sonar(self.data_loader,
                                                  model_submodule(self.model).keypoint_net,
                                                  noise_util=self.noise_util,
                                                  output_shape=params['res'],
                                                  top_k=params['top_k'],
                                                  conf_threshold=self.conf_threshold,
                                                  debug=self.debug,
                                                  use_color=use_color)
        if self.summary:
            self.summary["evaluation"][completed_epoch] = result_dict
        _print_result(result_dict)
