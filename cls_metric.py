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

        return

    def partial_compute_metric(self, dist_mat, gt_activity, pred_activity=None):
        for frame_cnt, loc_dist in enumerate(dist_mat):
            nb_active_gt = int(gt_activity[frame_cnt].sum())
            nb_active_pred = 2 if pred_activity is None else int(pred_activity[frame_cnt].sum()) #TODO remove hard coded max value of 2 DoAs 
            self._tp_doa += np.min((nb_active_gt, nb_active_pred))
            self._total_pred += nb_active_pred
            self._total_gt += nb_active_gt
            if nb_active_gt and nb_active_pred:
                if pred_activity is None:
                    if nb_active_gt==1:
                        loc_dist = loc_dist[:, 0][None]
                else:
                    loc_dist = loc_dist[pred_activity[frame_cnt]==1, :][:, gt_activity[frame_cnt]==1]
                row_ind, col_ind = linear_sum_assignment(loc_dist)
                loc_err = loc_dist[row_ind, col_ind].sum()

                self._localization_error += loc_err
        return

    def get_results(self):
        LE = self._localization_error/self._tp_doa
        LR = self._tp_doa/self._total_gt
        LP = self._tp_doa/self._total_pred
        LF = 2*LP*LR/(LP + LR + self._eps)
        return 180.*LE/np.pi, 100.*LR, 100.*LP, 100.*LF



