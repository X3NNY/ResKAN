
import argparse

import torch
from tqdm import trange
from model.ResKANet import ResKAN, ResCN, ResMLPN, ResFCN
from model.TCN import TCNModel
from model.CNN import CNNModel
from utils.plot import plot_loss, plot_result
from utils.results import calc_r2_pcc
from dataset.util import make_dataset
from model import *
from model.ResNet import ResNet, BasicBlock
import numpy as np
from config import lr, weight_decay, batch_size, n_wells, epochs, loss_wight
from utils.func import model_load, model_save, seed4everything, test_loss, train
# from utils.plot import plot_loss, plot_result


seed4everything(2024)


def determine_network(model_type: str = "ResKAN"):
    if model_type == "ResKAN":
        net_model = ResKAN([1, 8, 16, 16, 1], [3, 2, 1])
    elif model_type == "ResMLPN":
        net_model = ResMLPN([1, 8, 16, 16, 1], [3, 2, 1])
    elif model_type == "ResCN":
        net_model = ResCN([1, 8, 16, 16, 1], [3, 2, 1])
    elif model_type == "ResFCN":
        net_model = ResFCN([1, 8, 16, 16, 1], [3, 2, 1])
    elif model_type == "CNN":
        net_model = CNNModel()
    elif model_type == "TCN":
        net_model = TCNModel()
    elif model_type == "ResNet":
        net_model = ResNet(BasicBlock, 1)
    
    return net_model


def train(model_type: str):
    net_model = determine_network(model_type)

    train_loader, (x_val, y_val) = make_dataset(batch_size=batch_size, n_wells=n_wells)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net_model.parameters(),
                                 weight_decay=weight_decay,
                                 lr=lr)
    train_loss = [np.inf]
    val_loss = [np.inf]
    with trange(epochs) as t:
        for epoch in t:
            t.set_description(f"epoch: {epoch}")
            for x, y in train_loader:
                net_model.train()
                optimizer.zero_grad()
                y_pred = net_model(x)
                loss = loss_wight[0]*criterion(y_pred, y) + loss_wight[1]*net_model.loss()
                loss.backward()
                optimizer.step()
            train_loss.append(loss.item())
            if epoch % 20 == 0:
                with torch.no_grad():
                    net_model.eval()
                    y_pred = net_model(x_val)
                    vloss = criterion(y_pred, y_val)
            val_loss.append(vloss.item())
    
    plot_loss(train_loss, val_loss, names=['Training', 'Validation'], title=net_model.name, loss_type='MSE', save=True)
    plot_result(net_model)
    print("Test MSE loss: ", test_loss(net_model))
    print("r-squared and PCC: ", calc_r2_pcc(net_model))
    model_save(net_model)


if __name__ == "__main__":
    train("ResKAN")