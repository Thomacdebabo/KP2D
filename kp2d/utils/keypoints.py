# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch
from kp2d.datasets.noise_model import pol_2_cart,cart_2_pol
def warp_keypoints_old(keypoints, H):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]

def warp_keypoints(keypoints, H):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    # keypoints = pol_2_cart(torch.tensor(keypoints).unsqueeze(0), 60, 0.1, 5.0).squeeze(
    #     0).numpy()

    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    warped_points = warped_points[:, :2] / warped_points[:, 2:]
    warped_points = cart_2_pol(torch.tensor(warped_points).unsqueeze(0), 60, 0.1, 5.0).squeeze(0).numpy()
    return warped_points


def draw_keypoints(img_l, top_uvz, color=(255, 0, 0), idx=0):
    """Draw keypoints on an image"""
    vis_xyd = top_uvz.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()
    vis = img_l.copy()
    cnt = 0
    for pt in vis_xyd[:,:2].astype(np.int32):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x,y), 2, color, -1)
    return vis
