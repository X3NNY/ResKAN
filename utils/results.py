from dataset.util import marmousi_model, marmousi_seismic
import numpy as np
import torch
from scipy.stats import linregress

def calc_r2_pcc(
        model,
):
    AI = marmousi_model().T[:, 100:]
    seismic_offsets = np.expand_dims(marmousi_seismic().squeeze()[:, 100:], 1)
    seismic_offsets = torch.from_numpy((seismic_offsets - seismic_offsets.mean()) / seismic_offsets.std()).float()
    with torch.no_grad():
        model.cpu()
        model.eval()
        AI_inv = model(seismic_offsets).detach().numpy().squeeze()
    AI_inv = AI_inv * AI.std() + AI.mean()

    return r2_pcc_scores(AI, AI_inv)


def r2_pcc_scores(AI, AI_inv):
    """Function computes and prints the r2 and pcc on both the training and validation sets"""
    pcc_train = 0
    r2_train = 0
    for i in range(AI.shape[0]):
        trace_pred = AI_inv[i]
        trace_actual = AI[i]
        pcc_train += np.corrcoef(trace_actual, trace_pred)[0, 1]
        slope, intercept, r_value, p_value, std_err = linregress(trace_actual, trace_pred)
        r2_train += r_value ** 2
    pcc_train = pcc_train / AI.shape[0]
    r2_train = r2_train / AI.shape[0]

    return r2_train, pcc_train
