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

        self._fp_doa = 0
        self._fn_doa = 0
        self._ids = 0
        return

    def partial_compute_metric(self, dist_mat, gt_activity, pred_activity=None):
        if pred_activity is not None:
            M = pred_activity.sum(-1)
            N = gt_activity.sum(-1)
            self._fp_doa += (M-N).clip(min=0).sum(-1)
            self._fn_doa += (N-M).clip(min=0).sum(-1)
 
            self._ids += (pred_activity[1:]*(1-pred_activity[:-1])).sum(-1).sum(-1)

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
        MOTa = 1-(self._fp_doa + self._fn_doa + self._ids) / (self._total_gt + self._eps)
        LR = self._tp_doa/self._total_gt
        LP = self._tp_doa/self._total_pred
        LF = 2*LP*LR/(LP + LR + self._eps)
        return 180.*LE/np.pi, 100.*MOTa, self._ids, 100.*LR, 100.*LP, 100.*LF



