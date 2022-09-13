# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
from kp2d.utils.config import parse_train_file
from kp2d.utils.logging import printcolor, timing
from kp2d.utils.train_keypoint_net_utils import (_set_seeds, Trainer, parse_args)

import warnings
warnings.filterwarnings("ignore")





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

    printcolor('-'*25 + 'SINGLE GPU ' + '-'*25, 'cyan')
    
    if config.arch.seed is not None:
        _set_seeds(config.arch.seed)

    printcolor('-'*25 + ' MODEL PARAMS ' + '-'*25)
    printcolor(config.model.params, 'red')


    if config.datasets.augmentation.mode == "sonar_sim":
        warnings.warn('Sonar Simulator mode cannot be used in this script. Please use train_keypoint_net_sonar instead. default will be used instead.')

    trainer = Trainer(config)
    trainer.evaluation(0)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args.file)
