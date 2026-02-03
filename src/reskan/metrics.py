from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import linregress
from skimage.metrics import structural_similarity as ssim


def r2_pcc_scores(ai: np.ndarray, ai_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute mean r^2 and PCC over traces.
    Expects shapes: (N_traces, L) for both inputs.
    """
    stddevs = np.std(ai_pred, axis=1)
    valid_mask = stddevs != 0
    ai_pred = ai_pred[valid_mask]
    ai = ai[valid_mask]

    pcc = 0.0
    r2 = 0.0
    for i in range(ai.shape[0]):
        trace_pred = ai_pred[i]
        trace_actual = ai[i]
        pcc += float(np.corrcoef(trace_actual, trace_pred)[0, 1])
        r_value = linregress(trace_actual, trace_pred).rvalue
        r2 += float(r_value**2)

    pcc /= ai.shape[0]
    r2 /= ai.shape[0]
    return r2, pcc


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # In this project, impedance and prediction are typically scaled to [-1, 1]
    return float(ssim(img1, img2, data_range=2.0))


def calc_uiq(pred: np.ndarray, gt: np.ndarray, ws: int = 8) -> float:
    """
    Universal Image Quality Index (UIQ).
    """
    n = ws**2
    gt_sq = gt * gt
    pred_sq = pred * pred
    gt_pred = gt * pred

    gt_sum = uniform_filter(gt, ws)
    pred_sum = uniform_filter(pred, ws)
    gt_sq_sum = uniform_filter(gt_sq, ws)
    pred_sq_sum = uniform_filter(pred_sq, ws)
    gt_pred_sum = uniform_filter(gt_pred, ws)

    gt_pred_sum_mul = gt_sum * pred_sum
    gt_pred_sum_sq_sum_mul = gt_sum * gt_sum + pred_sum * pred_sum
    numerator = 4 * (n * gt_pred_sum - gt_pred_sum_mul) * gt_pred_sum_mul
    denominator1 = n * (gt_sq_sum + pred_sq_sum) - gt_pred_sum_sq_sum_mul
    denominator = denominator1 * gt_pred_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    idx = np.logical_and((denominator1 == 0), (gt_pred_sum_sq_sum_mul != 0))
    q_map[idx] = 2 * gt_pred_sum_mul[idx] / gt_pred_sum_sq_sum_mul[idx]
    idx = denominator != 0
    q_map[idx] = numerator[idx] / denominator[idx]

    s = int(np.round(ws / 2))
    return float(np.mean(q_map[s:-s, s:-s]))

