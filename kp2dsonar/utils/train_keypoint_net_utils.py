# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset, DataLoader

from kp2dsonar.datasets.augmentations import (ha_augment_sample, resize_sample,
                                              spatial_augment_sample,
                                              to_tensor_sample, to_tensor_sonar_sample, normalize_sample, a8x_normalize_sample)
from kp2dsonar.datasets.coco import COCOLoader
from kp2dsonar.datasets.patches_dataset import PatchesDataset
from kp2dsonar.datasets.sonarsim import SonarSimLoader


def sample_to_device(data, device):
    data['image'] = data['image'].to(device)
    data['image_aug'] = data['image_aug'].to(device)
    data['homography'] = data['homography'].to(device)
    return data


global _worker_init_fn
def _worker_init_fn(worker_id):
    """Worker init fn to fix the seed of the workers"""
    _set_seeds(42 + worker_id)


class image_transforms():
    def __init__(self, noise_util,config):
        self.angles = noise_util
        self.config = config
        mode = config.augmentation.mode
        self.transform = getattr(self,  "_" + mode)

    def _sonar_sim(self,sample):

        # sample = resize_sample(sample, image_shape=config.augmentation.image_shape)

        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        if self.noise_util.post_noise:
            sample = self.noise_util.add_noise_function(sample)
        sample = to_tensor_sonar_sample(sample)


        return sample

    def _sonar_real(self, sample): #TODO

        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        sample = to_tensor_sonar_sample(sample)

        return sample
    def _quantized_default(self, sample):

        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=self.config.augmentation.jittering)
        sample = a8x_normalize_sample(sample)

        return sample

    def _quantized_sonar(self, sample): #TODO: implement
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=self.config.augmentation.jittering)
        sample = a8x_normalize_sample(sample)

        return sample
    def _default(self, sample):

        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=self.config.augmentation.jittering)
        sample = normalize_sample(sample)

        return sample

    def __call__(self, sample):
        return self.transform(sample)

#Not yet in use
class image_transforms_eval():
    def __init__(self, noise_util,config):
        self.angles = noise_util
        self.config = config
        mode = config.augmentation.mode
        self.transform = getattr(self,  "_" + mode)

    def _sonar_sim(self,sample):

        sample = self.noise_util.pol_2_cart_sample(sample)
        sample = self.noise_util.augment_sample(sample)

        sample = self.noise_util.filter_sample(sample)
        sample = self.noise_util.cart_2_pol_sample(sample)
        if self.noise_util.post_noise:
            sample = self.noise_util.add_noise_function(sample)
        sample = to_tensor_sonar_sample(sample)


        return sample

    def _sonar_real(self, sample): #TODO
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = to_tensor_sample(sample)
        sample = normalize_sample(sample)

        return sample

    def _quantized_default(self, sample):
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = to_tensor_sample(sample)
        sample = a8x_normalize_sample(sample)

        return sample

    def _quantized_sonar(self, sample): #TODO: implement
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample)
        sample = a8x_normalize_sample(sample)
        return sample

    def _default(self, sample):
        sample = resize_sample(sample, image_shape=self.config.augmentation.image_shape)
        sample = to_tensor_sample(sample)
        sample = normalize_sample(sample)
        return sample

    def __call__(self, sample):
        return self.transform(sample)

def _set_seeds(seed=42):
    """Set Python random seeding and PyTorch seeds.
    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_datasets_and_dataloaders(config):
    """Prepare datasets for training, validation and test."""


    data_transforms = image_transforms(None, config)

    train_dataset = COCOLoader(config.train.path, data_transform=data_transforms)
    # Concatenate dataset to produce a larger one
    if config.train.repeat > 1:
        train_dataset = ConcatDataset([train_dataset for _ in range(config.train.repeat)])

    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              pin_memory=True,
                              num_workers=config.train.num_workers,
                              worker_init_fn=_worker_init_fn,
                              sampler=None,
                              drop_last=True)
    return train_dataset, train_loader

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

def setup_datasets_and_dataloaders_eval(config):
    """Prepare datasets for training, validation and test."""

    data_transforms = image_transforms_eval(None,config)



    hp_dataset = PatchesDataset(root_dir=config.val.path, use_color=True,
                                output_shape=config.augmentation.image_shape,
                                data_transform=None,
                                mode=config.augmentation.mode,
                                type='a')

    data_loader = DataLoader(hp_dataset,
                             batch_size=1,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=8,
                             worker_init_fn=None,
                             sampler=None,
                             drop_last=True)
    return hp_dataset, data_loader

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