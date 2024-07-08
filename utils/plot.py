from typing import List
import os

import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from dataset.util import marmousi_model, marmousi_seismic
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(*loss,
            names: List[str] = None,
            loss_type: str = '',
            title: str = '',
            save: bool = False):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, it in enumerate(loss):
        x = np.arange(0, 1001, 1001 // (len(it)-1))
        if names is not None:
            ax.plot(x, it, label=names[idx])
        else:
            ax.plot(x, it, label=f'loss_{idx}')
    ax.set_title(title)
    ax.set_xticks(np.arange(200, 1001, 200))
    ax.set_xlim(0, 1000)
    ax.set_ylabel(loss_type, fontsize=16)
    ax.set_xlabel('Epochs', fontsize=16)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.legend()
    plt.grid()

    if save == True:
        fig.savefig(f'results/img/{title}_{loss_type}.pdf', bbox_inches='tight')
    

def plot(img, cmap='rainbow', cbar_label=r'AI ($km/s\times g/cm^3$)', vmin=None, vmax=None):
    """Makes seaborn style plots"""
    dt = 0.00466
    dx = 6.25
    Y, X = np.mgrid[slice(0.47, 2.795+0.47 + dt, dt), slice(0, 17000 + dx, dx)]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    if (vmin is None or vmax is None):
        plt.pcolormesh(X, Y, img.T, cmap=cmap)
    else:
        plt.pcolormesh(X, Y, img.T, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar()
    plt.ylabel("Depth (Km)", fontsize=30)
    plt.xlabel("Distance (m)", fontsize=30, labelpad=15)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position("top")
    plt.gca().set_xticks(np.arange(0, 17000 + 1, 1700 * 2))
    plt.tick_params(axis='both', which='major', labelsize=30)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(cbar_label, rotation=270, fontsize=30, labelpad=40)
    return fig


def plot_result(model, save: bool = True):
    """Generate and save true and predicted AI plots"""
    AI = marmousi_model().T[:, 100:] / 1000
    seismic_offsets = np.expand_dims(marmousi_seismic().squeeze()[:, 100:], 1)
    seismic_offsets = torch.from_numpy((seismic_offsets - seismic_offsets.mean()) / seismic_offsets.std()).float()
    with torch.no_grad():
        model.cpu()
        model.eval()
        AI_inv = model(seismic_offsets).detach().numpy().squeeze()
    AI_inv = AI_inv * AI.std() + AI.mean()
    vmin = min([AI.min(), AI_inv.min()])
    vmax = max([AI.max(), AI_inv.max()])
    fig = plot(AI, vmin=vmin, vmax=vmax)
    if save:
        fig.savefig(f'results/img/ai/{model.name}_AI.png', bbox_inches='tight')
    fig = plot(AI_inv, vmin=vmin, vmax=vmax)
    if save:
        fig.savefig(f'results/img/ai/{model.name}_AI_inv.png', bbox_inches='tight')
    fig = plot(abs(AI_inv - AI), vmin=0, vmax=6, cmap='gray', cbar_label='Absolute Difference')
    if save:
        fig.savefig(f'results/img/ai/{model.name}_difference.png', bbox_inches='tight')