from model.ResKANet import ResKAN, ResCN, ResMLPN, ResFCN
from model.TCN import TCNModel
from model.CNN import CNNModel
from utils.plot import plot_result
from utils.results import calc_r2_pcc
from dataset.util import make_dataset
from model import *
from model.ResNet import ResNet, BasicBlock
import numpy as np
from utils.func import model_load, model_save, seed4everything, test_loss
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
        net_model = TCNModel(1, 1, [6, 10, 10, 10, 12, 12, 12], 5, 0.2)
    elif model_type == "ResNet":
        net_model = ResNet(BasicBlock, 1)
    
    model_load(net_model)

    return net_model


def test(model_type: str):
    net_model = determine_network(model_type)

    plot_result(net_model)
    print("Test MSE loss: ", test_loss(net_model))
    print("r-squared and PCC: ", calc_r2_pcc(net_model))


if __name__ == "__main__":
    test("ResKAN")
    test("CNN")
    test("TCN")
    test("ResNet")