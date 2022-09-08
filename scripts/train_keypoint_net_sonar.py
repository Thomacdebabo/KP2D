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

    printcolor('-'*25 + ' MODEL PARAMS ' + '-'*25)
    printcolor(config.model.params, 'red')

    # Setup model and datasets/dataloaders
    model = KeypointNetwithIOLoss(noise_util, mode=config.datasets.augmentation.mode,**config.model.params)
    train_dataset, train_loader = setup_datasets_and_dataloaders_sonar(config.datasets, noise_util)
    printcolor('({}) length: {}'.format("Train", len(train_dataset)))

    optimizer = optim.Adam(model.optim_params)

    # checkpoint model
    log_path = os.path.join(config.model.checkpoint_path, 'logs')
    os.makedirs(log_path, exist_ok=True)

    if not config.wandb.dry_run:
        summary = SummaryWriter(log_path,
                                config,
                                project=config.wandb.project,
                                entity=config.wandb.entity,
                                job_type='training',
                                mode=os.getenv('WANDB_MODE', 'run'))
        config.model.checkpoint_path = os.path.join(config.model.checkpoint_path, summary.run_name)
    else:
        summary = None
        date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        date_time = model_submodule(model).__class__.__name__ + '_' + date_time
        config.model.checkpoint_path = os.path.join(config.model.checkpoint_path, date_time)
    # added because when you run multiple jobs at once they sometimes overwrite each other
    i_dir = 1
    ending = ""
    while os.path.isdir(config.model.checkpoint_path + ending ):
        i_dir += 1
        ending = "_" + str(i_dir)
    config.model.checkpoint_path = config.model.checkpoint_path + ending


    print('Saving models at {}'.format(config.model.checkpoint_path))

    os.makedirs(config.model.checkpoint_path, exist_ok=True)


    # Initial evaluation
    evaluation(config, 0, model, summary,noise_util)
    # Train
    for epoch in range(config.arch.epochs):
        # train for one epoch (only log if eval to have aligned steps...)
        printcolor("\n--------------------------------------------------------------")
        train(config, train_loader, model, optimizer, epoch, summary)

        try:
            evaluation(config, epoch + 1, model, summary,noise_util)
        except:
            print("Evaluation failed...")
    printcolor('Training complete, models saved in {}'.format(config.model.checkpoint_path), "green")

def evaluation(config, completed_epoch, model, summary,noise_util):
    # Set to eval mode
    model.eval()
    model.training = False

    use_color = config.model.params.use_color

    eval_shape = config.datasets.augmentation.image_shape[::-1]
    eval_params = [{'res': eval_shape, 'top_k': 300}]
    for params in eval_params:
        # hp_dataset = SonarSimLoader(root_dir=config.datasets.val.path, noise_util=noise_util,
        #                             data_transform=image_transforms(noise_util, config.datasets)['train'])
        #
        # data_loader = DataLoader(hp_dataset,
        #                         batch_size=1,
        #                         pin_memory=False,
        #                         shuffle=False,
        #                         num_workers=8,
        #                         worker_init_fn=None,
        #                         sampler=None)
        hp_dataset, data_loader = setup_datasets_and_dataloaders_eval_sonar(config.datasets, noise_util)

        print('Loaded {} image pairs '.format(len(data_loader)))

        printcolor('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']))
        result_dict = evaluate_keypoint_net_sonar(data_loader,
                                                            model_submodule(model).keypoint_net,
                                                             noise_util,
                                                            output_shape=params['res'],
                                                            top_k=params['top_k'],
                                                            use_color=use_color)


        if summary:
            summary.add_scalar('repeatability_'+str(params['res']), result_dict['Repeatability'])
            summary.add_scalar('localization_' + str(params['res']), result_dict['Localization Error'])
            summary.add_scalar('correctness_'+str(params['res'])+'_'+str(1), result_dict['Correctness d1'])
            summary.add_scalar('correctness_'+str(params['res'])+'_'+str(5), result_dict['Correctness d5'])
            summary.add_scalar('correctness_'+str(params['res'])+'_'+str(10), result_dict['Correctness d10'])
            summary.add_scalar('mscore' + str(params['res']), result_dict['MScore'])

        _print_result(result_dict)


    # Save checkpoint
    if config.model.save_checkpoint:
        config['completed_epochs'] = completed_epoch
        current_model_path = os.path.join(config.model.checkpoint_path, 'model.ckpt')
        printcolor('\nSaving model (epoch:{}) at {}'.format(completed_epoch, current_model_path), 'green')
        torch.save(
        {
            'state_dict': model_submodule(model_submodule(model).keypoint_net).state_dict(),
            'config': config
        }, current_model_path)

def train(config, train_loader, model, optimizer, epoch, summary):
    # Set to train mode
    model.train()
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    # if args.adjust_lr:
    adjust_learning_rate(config, optimizer, epoch)

    pbar = tqdm(enumerate(train_loader, 0),
                unit=' images',
                unit_scale=config.datasets.train.batch_size,
                total=len(train_loader),
                smoothing=0,
                disable=False)
    running_loss = running_recall = grad_norm_disp = grad_norm_pose = grad_norm_keypoint = 0.0
    train_progress = float(epoch) / float(config.arch.epochs)

    log_freq = 10

    for (i, data) in pbar:

        # calculate loss
        optimizer.zero_grad()
        if config.device=='cpu':
            data_cuda = data
        else:
            data_cuda = sample_to_cuda(data)
        loss, recall = model(data_cuda)

        # compute gradient
        loss.backward()

        running_loss += float(loss)
        running_recall += recall

        # SGD step
        l = []
        for key,data in model.keypoint_net.state_dict().items():
            l.append(data.max().cpu().numpy())
        optimizer.step()
        # pretty progress bar
        pbar.set_description('Train [ E {}, T {:d}, R {:.4f}, R_Avg {:.4f}, L {:.4f}, L_Avg {:.4f}]'.format(
            epoch, epoch * config.datasets.train.repeat, recall, running_recall / (i + 1), float(loss),
            float(running_loss) / (i + 1)))

        i += 1
        if i % log_freq == 0:
            with torch.no_grad():
                if summary:
                    train_metrics = {
                        'train_loss': running_loss / (i + 1),
                        'train_acc': running_recall / (i + 1),
                        'train_progress': train_progress,
                    }

                    for param_group in optimizer.param_groups:
                        train_metrics[param_group['name'] + '_learning_rate'] = param_group['lr']

                    for k, v in train_metrics.items():
                        summary.add_scalar(k, v)

                    model(data_cuda, debug=True)
                    for k, v in model_submodule(model).vis.items():
                        summary.add_image(k, v)

if __name__ == '__main__':
    args = parse_args()
    main(args.file)
