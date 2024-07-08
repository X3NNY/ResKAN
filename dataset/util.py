from os.path import join as pjoin
import torch
from dataset.loader import SeismicDataset
import numpy as np
import segyio
from torch.utils.data import DataLoader


def marmousi_seismic():
    """Function returns the seismic data that comes with the marmousi model"""
    seismic = segyio.cube(pjoin('data', 'SYNTHETIC.segy'))
    seismic = np.transpose(seismic, axes=[1, 0, 2])
    return seismic


def marmousi_model():
    """Function returns the marmousi acoustic impedance model"""
    den_file = segyio.open(pjoin('data', 'MODEL_DENSITY_1.25m.segy'))
    rho = segyio.cube(den_file).squeeze().T
    rho = rho[::4, ::5]
    v_file = segyio.open(pjoin('data', 'MODEL_P-WAVE_VELOCITY_1.25m.segy'))
    vp = segyio.cube(v_file).squeeze().T
    vp = vp[::4, ::5]
    AI = vp * rho
    return AI


def train_val_split(n_wells: int):
    """Splits dataset into training and validation based on the number of well-logs specified by the user.

    The training traces are sampled uniformly along the length of the model. The validation data is all of the
    AI model except the training traces. Mean and Standard deviation are computed on the training data and used to
    standardize both the training and validation datasets.
    """
    # Load data
    seismic_offsets = marmousi_seismic().squeeze()[:, 100:]  # dim= No_of_gathers x trace_length
    impedance = marmousi_model().T[:, 100:]  # dim = No_of_traces x trace_length

    # Split into train and val
    train_indices = np.linspace(0, 2720, n_wells).astype(int)
    val_indices = np.setdiff1d(np.linspace(0, 2720, n_wells*10).astype(int), train_indices)
    x_train, y_train = seismic_offsets[train_indices], impedance[train_indices]
    x_val, y_val = seismic_offsets[val_indices], impedance[val_indices]

    # Standardize features and targets
    x_train_norm, y_train_norm = (x_train - x_train.mean()) / x_train.std(), (y_train - y_train.mean()) / y_train.std()
    x_val_norm, y_val_norm = (x_val - x_train.mean()) / x_train.std(), (y_val - y_train.mean()) / y_train.std()
    seismic_offsets = (seismic_offsets - x_train.mean()) / x_train.std()

    return x_train_norm, y_train_norm, x_val_norm, y_val_norm, seismic_offsets


def make_dataset(batch_size: int,
                n_wells: int,
                device: str = 'cuda'):
    x_train, y_train, x_val, y_val, seismic = train_val_split(n_wells)

    # Convert to torch tensors in the form (N, C, L)
    x_train = torch.from_numpy(np.expand_dims(x_train, 1)).float().to(device)
    y_train = torch.from_numpy(np.expand_dims(y_train, 1)).float().to(device)
    x_val = torch.from_numpy(np.expand_dims(x_val, 1)).float().to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, 1)).float().to(device)
    seismic = torch.from_numpy(np.expand_dims(seismic, 1)).float().to(device)

    # Set up the dataloader for training dataset
    dataset = SeismicDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False)
    
    return train_loader, (x_val, y_val)
