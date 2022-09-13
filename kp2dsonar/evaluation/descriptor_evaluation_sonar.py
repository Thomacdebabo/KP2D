# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/descriptor_evaluation.py

from kp2dsonar.utils.keypoints import warp_keypoints
from kp2dsonar.datasets.noise_model import pol_2_cart,cart_2_pol
import torch

#Pasted for debug
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

def convertToKeypoints(points):
    kps = []
    for p in points:
        kp = cv2.KeyPoint()
        kp.pt = tuple(p)
        kp.size = 1
        kps.append(kp)
    return kps
def visualizeMatches(img1, kp1, img2, kp2, matches, save= True, prefix = "", path = '', fname = ""):

    kp1 = convertToKeypoints(kp1)

    kp2 = convertToKeypoints(kp2)


    train_kp_image = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    plt.figure()
    plt.imshow(train_kp_image)
    match_img = cv2.drawMatches(img2, kp2, img1, kp1, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(match_img)
    plt.show()
    if save:
        cv2.imwrite(os.path.join(path, prefix+"train_kp_image" + fname+".jpg"),  train_kp_image)
        cv2.imwrite(os.path.join(path,  prefix+"match_img" + fname+".jpg"), match_img)

def select_k_best(points, descriptors, k):
    """ Select the k most probable points (and strip their probability).
    points has shape (num_points, 3) where the last coordinate is the probability.

    Parameters
    ----------
    points: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    k: int
        Number of keypoints to select, based on probability.
    Returns
    -------
    
    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    sorted_prob = points[points[:, 2].argsort(), :2]
    sorted_desc = descriptors[points[:, 2].argsort(), :]

    start = min(k, points.shape[0])
    selected_points = sorted_prob[-start:, :]
    selected_descriptors = sorted_desc[-start:, :]
    return selected_points, selected_descriptors

def normalize_keypoints(kp, f , a):
    return kp / f - a

def unnormalize_keypoints(kp, f , a):
    return (kp + a)*f

def keep_shared_points(keypoints, descriptors, H, shape, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    
    Parameters
    ----------
    keypoints: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    H: numpy.ndarray (3,3)
        Homography.
    shape: tuple 
        Image shape.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    
    def keep_true_keypoints(points, descriptors, H, shape):
        """ Keep only the points whose warped coordinates by H are still inside shape. """
        warped_points = warp_keypoints(points[:,:2], H)
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :], descriptors[mask, :]

    selected_keypoints, selected_descriptors = keep_true_keypoints(keypoints, descriptors, H, shape)
    selected_keypoints, selected_descriptors = select_k_best(selected_keypoints, selected_descriptors, keep_k_points)
    return selected_keypoints, selected_descriptors



def compute_matching_score_sonar(data, keep_k_points=1000):
    """
    Compute the matching score between two sets of keypoints with associated descriptors.
    
    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
        desc: numpy.ndarray (N,256)
            Keypoint descriptors.
        warped_desc: numpy.ndarray (N,256)
            Warped keypoint descriptors.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    ms: float
        Matching score.
    """
    shape = data['image_shape']
    real_H = data['homography']
    sonar_config = data['sonar_config']

    f = [shape[0] / 2, shape[1] / 2]
    a = [1,1]
    # Filter out predictions
    keypoints = data['prob']
    warped_keypoints = data['warped_prob']

    desc = data['desc']
    warped_desc = data['warped_desc']
    sonar_config = data['sonar_config']
    
    # Keeps all points for the next frame. The matching for caculating M.Score shouldnt use only in view points.
    keypoints,        desc        = select_k_best(keypoints,               desc, keep_k_points)
    warped_keypoints, warped_desc = select_k_best(warped_keypoints, warped_desc, keep_k_points)

    cart_keypoints = normalize_keypoints(keypoints.copy() ,f,a)
    cart_keypoints = torch.tensor(cart_keypoints)
    cart_keypoints = pol_2_cart(cart_keypoints.unsqueeze(0),
                                sonar_config["fov"],
                                sonar_config["r_min"],
                                sonar_config["r_max"]).squeeze(0).numpy()

    cart_warped_keypoints = normalize_keypoints(warped_keypoints.copy(), f, a)
    cart_warped_keypoints = torch.tensor(cart_warped_keypoints)
    cart_warped_keypoints = pol_2_cart(cart_warped_keypoints.unsqueeze(0),
                                       sonar_config["fov"],
                                       sonar_config["r_min"],
                                       sonar_config["r_max"]).squeeze(0).numpy()

    
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    # This part needs to be done with crossCheck=False.
    # All the matched pairs need to be evaluated without any selection.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.match(desc, warped_desc)
    if not matches:
        return 0
    matches_idx = np.array([m.queryIdx for m in matches])

    m_keypoints = cart_keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = cart_warped_keypoints[matches_idx, :]

    true_warped_keypoints = warp_keypoints(m_keypoints, np.linalg.inv(real_H))
    true_warped_keypoints = cart_2_pol(torch.tensor(true_warped_keypoints).unsqueeze(0), sonar_config["fov"], sonar_config["r_min"], sonar_config["r_max"]).squeeze(0).numpy()
    true_warped_keypoints = unnormalize_keypoints(true_warped_keypoints,f,a)

    keypoints_warped_pol = cart_2_pol(torch.tensor(m_warped_keypoints).unsqueeze(0), sonar_config["fov"], sonar_config["r_min"], sonar_config["r_max"]).squeeze(0).numpy()
    keypoints_warped_pol = unnormalize_keypoints(keypoints_warped_pol,f,a)

    vis_warped = np.all((true_warped_keypoints >= 0) & (true_warped_keypoints <= (np.array(shape)-1)), axis=-1)
    norm1 = np.linalg.norm(true_warped_keypoints - keypoints_warped_pol, axis=-1)

    correct1 = (norm1 < 3)
    count1 = np.sum(correct1 * vis_warped)
    score1 = count1 / np.maximum(np.sum(vis_warped), 1.0)

    matches = bf.match(warped_desc, desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_warped_keypoints = cart_warped_keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_keypoints = cart_keypoints[matches_idx, :]

    true_keypoints = warp_keypoints(m_warped_keypoints, real_H)
    true_keypoints = cart_2_pol(torch.tensor(true_keypoints).unsqueeze(0), 60, 0.1, 5.0).squeeze(0).numpy()
    true_keypoints = unnormalize_keypoints(true_keypoints,f,a)

    keypoints_pol = cart_2_pol(torch.tensor(m_keypoints).unsqueeze(0), 60, 0.1, 5.0).squeeze(0).numpy()
    keypoints_pol = unnormalize_keypoints(keypoints_pol,f,a)

    vis = np.all((true_keypoints >= 0) & (true_keypoints <= (np.array(shape)-1)), axis=-1)
    norm2 = np.linalg.norm(true_keypoints - keypoints_pol, axis=-1)

    correct2 = (norm2 < 3)
    count2 = np.sum(correct2 * vis)
    score2 = count2 / np.maximum(np.sum(vis), 1.0)

    ms = (score1 + score2) / 2
    return ms

def compute_homography_sonar(data, noise_util, keep_k_points=1000, debug = False):
    """
    Compute the homography between 2 sets of Keypoints and descriptors inside data. 
    Use the homography to compute the correctness metrics (1,3,5).

    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
        desc: numpy.ndarray (N,256)
            Keypoint descriptors.
        warped_desc: numpy.ndarray (N,256)
            Warped keypoint descriptors.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    correctness1: float
        correctness1 metric.
    correctness3: float
        correctness3 metric.
    correctness5: float
        correctness5 metric.
    """
    shape = data['image'].shape[1:]
    real_H = data['homography']
    sonar_config = data['sonar_config']

    f = [shape[0]/2, shape[1]/2]
    a = [1,1]

    f3 = [shape[0]/2, shape[1]/2, 1]
    a3 = [1,1,0]

    keypoints = data['prob']
    warped_keypoints = data['warped_prob']
    desc = data['desc']
    warped_desc = data['warped_desc']

    # Keeps only the points shared between the two views
    cart_keypoints = torch.tensor(keypoints.copy()/f3-a3)
    cart_keypoints = pol_2_cart(cart_keypoints.unsqueeze(0), sonar_config["fov"], sonar_config["r_min"], sonar_config["r_max"]).squeeze(
        0).numpy()
    def keep_true_keypoints(points,warped_points, descriptors,):
        mask = (warped_points[:, 0] >= -1) & (warped_points[:, 0] < 1) & \
               (warped_points[:, 1] >= -1) & (warped_points[:, 1] < 1)
        return points[mask, :], descriptors[mask, :]

    ckw = warp_keypoints(cart_keypoints[:,:2], np.linalg.inv(real_H))
    ckw = cart_2_pol(torch.tensor(ckw).unsqueeze(0), sonar_config["fov"], sonar_config["r_min"], sonar_config["r_max"]).squeeze(0).numpy()
    cart_keypoints, desc = keep_true_keypoints(cart_keypoints, ckw, desc)
    cart_keypoints, desc = select_k_best(cart_keypoints, desc, keep_k_points)

    cart_warped_keypoints = torch.tensor(warped_keypoints.copy() / f3 - a3)
    cart_warped_keypoints = pol_2_cart(cart_warped_keypoints.unsqueeze(0), sonar_config["fov"], sonar_config["r_min"], sonar_config["r_max"]).squeeze(
        0).numpy()

    ckw_2 = warp_keypoints(cart_warped_keypoints[:,:2], real_H)
    ckw_2 = cart_2_pol(torch.tensor(ckw_2).unsqueeze(0), sonar_config["fov"], sonar_config["r_min"], sonar_config["r_max"]).squeeze(0).numpy()
    cart_warped_keypoints, warped_desc = keep_true_keypoints(cart_warped_keypoints, ckw_2, warped_desc)
    cart_warped_keypoints, warped_desc = select_k_best(cart_warped_keypoints, warped_desc, keep_k_points)



    try:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc, warped_desc)
        matches_idx = np.array([m.queryIdx for m in matches])
        m_keypoints = cart_keypoints[matches_idx, :]
    except:
        return 0,0,0,0,0,0
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = cart_warped_keypoints[matches_idx, :]
    if m_keypoints.shape[0] <4 or m_warped_keypoints.shape[0] <4:
        return 0,0,0,0,0,0

    # Estimate the homography between the matches using RANSAC
    #This has to be done unnormalized for some reason?
    H, h_mask = cv2.findHomography(unnormalize_keypoints(m_warped_keypoints, f, a), unnormalize_keypoints(m_keypoints, f, a), cv2.RANSAC, 5.0)


    if H is None:
        return 0, 0, 0,0,0,0

    #BEWARE: apparently computing homographies in normalized coordinates does not work? So we have to estimate H in unnormalized coordinates.
    #This leads to us having to treat H and real_H differently
    corners = np.array([[shape[0]/2, 0],
                        [0, shape[1] - 1],
                        [shape[0] - 1, shape[1] - 1],
                        [shape[0]/2, shape[1]-1]])
    corners = normalize_keypoints(corners, f, a)

    real_warped_corners = warp_keypoints(corners, real_H)
    real_warped_corners = unnormalize_keypoints(real_warped_corners, f, a)

    warped_corners = warp_keypoints(unnormalize_keypoints(corners, f, a),H)
    estimated_warped_keypoints = warp_keypoints(unnormalize_keypoints(m_warped_keypoints, f, a), H)

    real_warped_keypoints = warp_keypoints(m_keypoints, np.linalg.inv(real_H))
    real_warped_keypoints = unnormalize_keypoints(real_warped_keypoints, f, a)

    warped_keypoints = warp_keypoints(m_warped_keypoints, real_H)
    warped_keypoints = unnormalize_keypoints(warped_keypoints, f, a)

    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness1 = float(mean_dist <= 1)
    correctness5 = float(mean_dist <= 5)
    correctness10 = float(mean_dist <= 10)
    useful_points_ratio = float(h_mask.sum()/matches_idx.__len__())
    abs_points = int(h_mask.sum())
    mean_distance = float(mean_dist)

    #DEBUG PICS
    if debug:
        img_debug = noise_util.pol_2_cart_torch(torch.tensor(data['image'].copy()/255).unsqueeze(0))
        img_debug = norm_img(img_debug)
        img_debug = np.ascontiguousarray((img_debug*255).squeeze(0).numpy().astype(np.uint8).transpose(1,2,0))

        img_debug = draw_kps(img_debug, unnormalize_keypoints(corners,f,a), c = (0,0,255))
        img_debug = draw_kps(img_debug, real_warped_corners, c = (255,0,255))
        #img_debug = draw_kps(img_debug, m_warped_keypoints, c = (255,0,0))
        img_debug = draw_kps(img_debug,  unnormalize_keypoints(m_keypoints,f,a), c = (0,255,0))
        img_debug = draw_kps(img_debug, warped_keypoints, c = (255,0,0))

        cv2.imshow("hi", img_debug)

        img_debug = noise_util.pol_2_cart_torch(torch.tensor(data['image'].copy() / 255).unsqueeze(0))
        img_debug = norm_img(img_debug)
        img_debug = np.ascontiguousarray((img_debug * 255).squeeze(0).numpy().astype(np.uint8).transpose(1, 2, 0))
        query_image = img_debug.copy()

        img_debug = draw_kps(img_debug, warped_corners , c=(0, 0, 255))
        img_debug = draw_kps(img_debug, real_warped_corners, c=(255, 0, 255))
        img_debug = draw_kps(img_debug, unnormalize_keypoints(m_keypoints, f, a), c=(0, 255, 0))
        img_debug = draw_kps(img_debug, estimated_warped_keypoints, c=(255, 0, 0))

        cv2.imshow("hi3", img_debug)

        img_debug = noise_util.pol_2_cart_torch(torch.tensor(data['image_aug'].copy()/255).unsqueeze(0))
        img_debug = norm_img(img_debug)
        img_debug = np.ascontiguousarray((img_debug*255).squeeze(0).numpy().astype(np.uint8).transpose(1,2,0))
        trainImage = img_debug.copy()

        img_debug = draw_kps(img_debug, unnormalize_keypoints(m_warped_keypoints, f, a), c = (255,0,0))
        img_debug = draw_kps(img_debug, real_warped_keypoints, c = (0,255,0))

        cv2.imshow("hi2", img_debug)


        cv2.waitKey(1)
        # try:
        #     visualizeMatches(trainImage, unnormalize_keypoints(cart_warped_keypoints, f, a), query_image, unnormalize_keypoints(cart_keypoints, f, a), matches)
        # except:
        #     print("whopsie")

    return correctness1, correctness5, correctness10, useful_points_ratio, mean_distance, abs_points

def norm_img(img):
    return (img - img.min())/img.max()

def draw_kps(img_debug, corners, c = (0, 0, 255)):
    for pt in corners[:,:2].astype(np.int32):
        x, y = int(pt[0]), int(pt[1])
        img_debug = cv2.circle(img_debug, (x,y), 2,  c, -1)
    return img_debug
