# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
from kp2dsonar.utils.config import parse_train_file
from kp2dsonar.utils.logging import printcolor
from kp2dsonar.utils.train_keypoint_net_utils_sonar import TrainerSonar
from kp2dsonar.utils.train_keypoint_net_utils import _set_seeds,parse_args


def _print_result(result_dict):
    for k in result_dict.keys():
        print("%s: %.3f" %( k, result_dict[k]))



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
