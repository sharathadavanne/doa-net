import numpy as np
import random
import pickle
from IPython import  embed
eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment

def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section
    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost -  distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """
    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat, da_mat = np.zeros((gt_len, pred_len)), np.zeros((gt_len, pred_len), dtype=int)

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat/180.)
    da_mat[row_ind, col_ind] = 1
    return cost_mat, da_mat


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    #### MAIN ALGO starts here
    nb_train_samples = 1000
    nb_test_samples = int(0.1*1000)
    pickle_filename = 'hung_data'

    max_doas = 5
    doas_range = range(1,max_doas)
    azi_range = range(-180, 180, 1)
    ele_range = range(-60, 61, 1)
    for set_cnt, nb_samples in enumerate([nb_train_samples, nb_test_samples]):
        data_dict = {}
        for cnt in range(nb_samples):
            nb_ref = random.choice(doas_range)
            nb_pred = random.choice(doas_range)

            ref_ang = np.array((random.sample(azi_range, nb_ref), random.sample(ele_range, nb_ref))).T
            pred_ang = np.array((random.sample(azi_range, nb_pred), random.sample(ele_range, nb_pred))).T

            cost_mat, da_mat = least_distance_between_gt_pred(ref_ang, pred_ang)
            data_dict[cnt] = [nb_ref, nb_pred, cost_mat, da_mat]

        extention = 'test' if set_cnt else 'train'
        out_filename = 'data/{}_{}'.format(pickle_filename, extention)
        print('Saving data in: {}'.format(out_filename))
        save_obj(data_dict, out_filename)


if __name__ == "__main__":
    main()
