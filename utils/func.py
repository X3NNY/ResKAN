import io
import os
import pickle
import random
import torch
import torch.nn
from config import loss_wight
from dataset.util import marmousi_model, marmousi_seismic
import numpy as np
from tqdm import trange

def seed4everything(seed=1997):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_save(
    model: torch.nn.Module,
):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    res = {
        "model": buffer,
    }
    with open(os.path.join("results", "model", f"{model.name}.pkl",), "wb") as file:
        pickle.dump(res, file)


def model_load(
    model: torch.nn.Module,
    device: str = 'cuda'
):
    with open(os.path.join("results", "model", f"{model.name}.pkl"), "rb") as file:
        res = pickle.load(file)
    
    res["model"].seek(0)

    model.load_state_dict(torch.load(res["model"]))
    model.to(device)


def test_loss(
        model,
    ):
    criterion = torch.nn.MSELoss()
    
    AI = np.expand_dims(marmousi_model().T[:, 100:].squeeze(), 1)
    AI = torch.from_numpy((AI - AI.mean()) / AI.std()).float()
    seismic_offsets = np.expand_dims(marmousi_seismic().squeeze()[:, 100:], 1)
    seismic_offsets = torch.from_numpy((seismic_offsets - seismic_offsets.mean()) / seismic_offsets.std()).float()
    with torch.no_grad():
        model.cpu()
        model.eval()
        AI_inv = model(seismic_offsets)
        loss = criterion(AI_inv, AI)
        return loss.item()