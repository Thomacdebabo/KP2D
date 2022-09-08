# Copyright 2020 Toyota Research Institute.  All rights reserved.
import warnings
import argparse
import os
from datetime import datetime

import torch
import torch.optim as optim

from tqdm import tqdm
import json
from kp2dsonar.evaluation.evaluate import evaluate_keypoint_net
from kp2dsonar.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2dsonar.utils.config import parse_train_file
from kp2dsonar.utils.logging import printcolor
from kp2dsonar.utils.train_keypoint_net_utils import (_set_seeds, sample_to_device,
                                      setup_datasets_and_dataloaders, setup_datasets_and_dataloaders_eval)

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

class Trainer:
    """
    This class implements a simple training pipeline. It makes the code easier to read and more structured.
    """
    def __init__(self, config):
        self.model = KeypointNetwithIOLoss(mode='default', **config.model.params)
        self.optimizer = optim.Adam(self.model.optim_params)
        self.init_datasets(config)
        self.summary = {"evaluation": {}, "train": {}}
        self.config = config
        self.eval_params = [{'res': self.config.datasets.augmentation.image_shape[::-1], 'top_k': 300}]

        self.init_dir()
        printcolor('({}) length: {}'.format("Train", len(self.train_dataset)))
    def init_datasets(self, config):
        self.train_dataset, self.train_loader = setup_datasets_and_dataloaders(config.datasets)
        self.hp_dataset, self.data_loader = setup_datasets_and_dataloaders_eval(config.datasets)

    def init_dir(self):
        date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        date_time = model_submodule(self.model).__class__.__name__ + '_' + date_time
        self.config.model.checkpoint_path = os.path.join(self.config.model.checkpoint_path,
                                                    self.config.model.params.keypoint_net_type + "_" + date_time)
        # added because when you run multiple jobs at once they sometimes overwrite each other
        i_dir = 1
        ending = ""
        while os.path.isdir(self.config.model.checkpoint_path + ending):
            i_dir += 1
            ending = "_" + str(i_dir)
        self.config.model.checkpoint_path = self.config.model.checkpoint_path + ending
        print('Saving models at {}'.format(self.config.model.checkpoint_path))

        os.makedirs(self.config.model.checkpoint_path, exist_ok=True)

    def evaluation(self, completed_epoch):
        # Set to eval mode
        self.model.eval()
        self.model.training = False

        use_color = self.config.model.params.use_color



        for params in self.eval_params:

            print('Loaded {} image pairs '.format(len(self.data_loader)))

            printcolor('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']))
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(self.data_loader,
                                                                model_submodule(self.model).keypoint_net,
                                                                output_shape=params['res'],
                                                                top_k=params['top_k'],
                                                                use_color=use_color)
            if self.summary:
                self.summary["evaluation"][completed_epoch] = {'Repeatability':rep,
                                                          'Localization Error':loc,
                                                          'Correctness d1':c1,
                                                          'Correctness d3':c3,
                                                          'Correctness d5':c5,
                                                          'MScore':mscore}

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))

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

    def train_single_epoch(self, epoch, log_freq = 1000):
        # Set to train mode
        self.model.train()
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        # if args.adjust_lr:
        adjust_learning_rate(self.config, self.optimizer, epoch)

        pbar = tqdm(enumerate(self.train_loader, 0),
                    unit=' images',
                    unit_scale=self.config.datasets.train.batch_size,
                    total=len(self.train_loader),
                    smoothing=0,
                    disable=False)
        running_loss = 0.0
        running_recall = 0.0
        train_progress = float(epoch) / float(self.config.arch.epochs)

        running_summary = {}
        for (i, data) in pbar:

            # calculate loss
            self.optimizer.zero_grad()
            data_cuda = sample_to_device(data, self.config.device)
            loss, recall = self.model(data_cuda)

            # compute gradient
            loss.backward()

            running_loss += float(loss)
            running_recall += recall

            # SGD step
            l = []
            for key,data in self.model.keypoint_net.state_dict().items():
                l.append(data.max().cpu().numpy())
            self.optimizer.step()
            # pretty progress bar
            pbar.set_description('Train [ E {}, T {:d}, R {:.4f}, R_Avg {:.4f}, L {:.4f}, L_Avg {:.4f}]'.format(
                epoch, epoch * self.config.datasets.train.repeat, recall, running_recall / (i + 1), float(loss),
                float(running_loss) / (i + 1)))

            i += 1
            if i % log_freq == 0:
                if self.summary:
                    train_metrics = {
                        'train_loss': running_loss / (i + 1),
                        'train_acc': running_recall / (i + 1),
                        'train_progress': train_progress,
                    }
                    running_summary[i] = train_metrics

        self.summary["train"][epoch] = running_summary

    def train(self):

        for epoch in range(self.config.arch.epochs):
            # train for one epoch (only log if eval to have aligned steps...)
            printcolor("\n--------------------------------------------------------------")
            self.train_single_epoch(epoch)

            try:
                self.evaluation(epoch + 1)
            except:
                print("Evaluation failed...")

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
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args.file)
