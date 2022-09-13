# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
from tqdm import tqdm

from kp2dsonar.evaluation.descriptor_evaluation_sonar import (compute_homography_sonar,
                                                         compute_matching_score_sonar)
from kp2dsonar.evaluation.detector_evaluation_sonar import compute_repeatability_sonar
from kp2d.utils.image import  to_gray_normalized
from kp2dsonar.datasets.noise_model import to_numpy

def evaluate_keypoint_net_sonar(data_loader, keypoint_net, noise_util, output_shape=(512, 512),conf_threshold = 0.9, top_k=300,
                          use_color=True, device = 'cpu', debug = False):
    """Keypoint net evaluation script.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False


    localization_err, repeatability = [], []
    correctness1, correctness5, correctness10, useful_points,absolute_amt_points, mean_distance, MScore,p_amt = [], [], [], [], [], [], [], []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):

            image = sample['image'].to(keypoint_net.device)
            warped_image =sample['image_aug'].to(keypoint_net.device)

            score_1, coord_1, desc1 = keypoint_net(image)
            score_2, coord_2, desc2 = keypoint_net(warped_image)
            B, C, Hc, Wc = desc1.shape

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(C, -1).t().cpu().numpy()
            desc2 = desc2.view(C, -1).t().cpu().numpy()

            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]
            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape': output_shape,
                    'image_aug': sample['image_aug'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1,
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2,
                    'sonar_config': vars(noise_util)}

            # Compute repeatabilty and localization error
            N1, N2, rep, loc_err = compute_repeatability_sonar(data, keep_k_points=top_k,
                                                             distance_thresh=3)

            N = float((N1+N2)/2)

            if (rep != -1) and (loc_err != -1):
                repeatability.append(rep)
                localization_err.append(loc_err)
                p_amt.append(N)


            # Compute correctness
            c1, c5, c10, up, md, ap = compute_homography_sonar(data,noise_util, keep_k_points=top_k,debug=debug) #TODO remove noise util once debugging is done

            correctness1.append(c1)
            correctness5.append(c5)
            correctness10.append(c10)
            useful_points.append(up)
            mean_distance.append(md)
            absolute_amt_points.append(ap)

            # Compute matching score
            mscore = compute_matching_score_sonar(data, keep_k_points=top_k)
            MScore.append(mscore)

    return {'Repeatability': np.mean(repeatability).item(),
     'Localization Error': np.mean(localization_err).item(),
     'Amount of good points': np.mean(p_amt).item(),
     'Correctness d1': np.mean(correctness1).item(),
     'Correctness d5': np.mean(correctness5).item(),
     'Correctness d10': np.mean(correctness10).item(),
     'MScore': np.mean(MScore).item(),
     'Useful points ratio ': np.mean(useful_points).item(),
     'Absolute amount of used points ': np.mean(absolute_amt_points).item(),
     'Mean distance (debug)': np.mean(mean_distance).item()}


def evaluate_ORB_sonar(data_loader, detector, noise_util, output_shape=(512, 512), top_k=300,
                          use_color=True, device = 'cuda'):
    """Keypoint net evaluation script.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """


    conf_threshold = 0.9
    localization_err, repeatability = [], []
    correctness1, correctness5, correctness10, useful_points,absolute_amt_points, mean_distance, MScore,p_amt = [], [], [], [], [], [], [], []
    no_points = 0
    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
                image = to_numpy(sample['image'])
                warped_image = to_numpy(sample['image_aug'])
            else:
                image = to_gray_normalized(sample['image']).numpy()
                warped_image = to_gray_normalized(sample['image_aug']).numpy()
            coord_1, desc1 = detector.detectAndCompute(image, None)
            coord_2, desc2 = detector.detectAndCompute(warped_image, None)

            if desc1 is None or desc2 is None:
                print("no pointies found")
                no_points += 1
                continue

            t = []
            for c1 in coord_1:
                # kps_list.append([(kp.pt[0]-256)/256,kp.pt[1]/440-1])
                t.append([c1.pt[0],c1.pt[1]])

            coord_1 = np.array(t)

            t = []
            for c2 in coord_2:
                # kps_list.append([(kp.pt[0]-256)/256,kp.pt[1]/440-1])
                t.append([c2.pt[0],c2.pt[1]])
            coord_2 = np.array(t)
            desc1 = np.array(desc1)
            desc2 = np.array(desc2)



            # Scores & Descriptors
            score_1 = np.concatenate([coord_1, (np.ones((coord_1.shape[0],1)))], axis=1)
            score_2 = np.concatenate([coord_2, np.ones((coord_2.shape[0],1))], axis=1)


            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]
            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape': output_shape,
                    'image_aug': sample['image_aug'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1,
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2,
                    'sonar_config': vars(noise_util)}

            # Compute repeatabilty and localization error
            N1, N2, rep, loc_err = compute_repeatability_sonar(data, keep_k_points=top_k,
                                                             distance_thresh=3)

            N = float((N1+N2)/2)

            if (rep != -1) and (loc_err != -1):
                repeatability.append(rep)
                localization_err.append(loc_err)
                p_amt.append(N)


            # Compute correctness
            c1, c5, c10, up, md, ap = compute_homography_sonar(data,noise_util, keep_k_points=top_k) #TODO remove noise util once debugging is done

            correctness1.append(c1)
            correctness5.append(c5)
            correctness10.append(c10)
            useful_points.append(up)
            mean_distance.append(md)
            absolute_amt_points.append(ap)

            # Compute matching score
            mscore = compute_matching_score_sonar(data, keep_k_points=top_k)
            MScore.append(mscore)

    return {'Repeatability': np.mean(repeatability).item(),
         'Localization Error': np.mean(localization_err).item(),
         'Amount of good points': np.mean(p_amt).item(),
         'Correctness d1': np.mean(correctness1).item(),
         'Correctness d5': np.mean(correctness5).item(),
         'Correctness d10': np.mean(correctness10).item(),
         'MScore': np.mean(MScore).item(),
         'Useful points ratio ': np.mean(useful_points).item(),
         'Absolute amount of used points ': np.mean(absolute_amt_points).item(),
         'Mean distance (debug)': np.mean(mean_distance).item()}
