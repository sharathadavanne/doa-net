from scipy.optimize import linear_sum_assignment
import numpy as np
from IPython import embed


class doa_metric:
    def __init__(self):
        self._eps = 1e-7

        self._localization_error = 0
        self._total_gt = self._eps

        self._tp_doa = 0
        self._total_pred = self._eps

        self._is_baseline = False
        return

    def partial_compute_metric(self, dist_mat, gt_activity, pred_activity=None):
        if pred_activity is None:
            self._is_baseline = True
            for frame_cnt, loc_dist in enumerate(dist_mat):
                nb_active = int(gt_activity[frame_cnt].sum())
                if nb_active:
                    if nb_active == 1:
                        loc_dist = loc_dist[:, 0][None]

                    row_ind, col_ind = linear_sum_assignment(loc_dist)
                    loc_err = loc_dist[row_ind, col_ind].sum()

                    self._total_gt += nb_active
                    self._localization_error += loc_err
        else:
            for frame_cnt, loc_dist in enumerate(dist_mat):
                nb_active_gt = int(gt_activity[frame_cnt].sum())
                nb_active_pred = int(pred_activity[frame_cnt].sum())
                self._tp_doa += np.min((nb_active_gt, nb_active_pred))
                self._total_pred += nb_active_pred
                self._total_gt += nb_active_gt
                if nb_active_gt and nb_active_pred:
                    loc_dist = loc_dist[pred_activity[frame_cnt]==1, :][:, gt_activity[frame_cnt]==1]
                    row_ind, col_ind = linear_sum_assignment(loc_dist)
                    loc_err = loc_dist[row_ind, col_ind].sum()

                    self._localization_error += loc_err
        return

    def get_results(self):
        if self._is_baseline:
            return self._localization_error/self._total_gt
        else:
            localization_error = self._localization_error/self._tp_doa
            localization_recall = self._tp_doa/self._total_gt
            return localization_error, localization_recall



