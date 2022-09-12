# Copyright 2020 Toyota Research Institute.  All rights reserved.
def to_tensor_sonar_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """

    sample['image'] =  sample['image'].type(tensor_type).div(255.0)
    sample['image_aug'] =  sample['image_aug'].type(tensor_type).div(255.0)
    return sample




