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


def train(
        model: torch.nn.Module,
        n_epoch: int,
        train_loader,
        val,
        lr: float = 0.001,
        weight_decay: float = 1e-4):
    print("Parameters count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up loss
    criterion = torch.nn.MSELoss()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=weight_decay,
                                 lr=lr)

    # Set up list to store the losses
    train_loss = [np.inf]
    val_loss = [np.inf]
    # Start training
    with trange(n_epoch) as t:
        for epoch in t:
            t.set_description(f"epoch: {epoch}")
            for x, y in train_loader:
                model.train()
                optimizer.zero_grad()
                y_pred = model(x)
                
                loss = loss_wight[0] * criterion(y_pred, y) + loss_wight[1] * model.loss()
                loss.backward()
                optimizer.step()
            train_loss.append(loss.item())
            if epoch % 20 == 0:
                with torch.no_grad():
                    model.eval()
                    y_pred = model(val[0])
                    vloss = criterion(y_pred, val[1])
            val_loss.append(vloss.item())
            t.set_postfix_str(
                f'Training loss: {train_loss[-1]:0.4f} | Validation loss: {val_loss[-1]:0.4f}'
            )
    
    return train_loss, val_loss

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