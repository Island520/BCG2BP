import torch
import numpy as np
import torch.nn.functional as F


def scale_F_BCG(F_BCG_original, current_dbp, current_sbp, ref_BCG):
    if len(F_BCG_original) == 0:
        return F_BCG_original  # Return the empty array as is

    min_val = torch.min(ref_BCG)
    max_val = torch.max(ref_BCG)

    F_BCG_scaled = ((F_BCG_original - min_val) / (max_val - min_val + 1e-8)) * (current_sbp - current_dbp) + current_dbp

    return F_BCG_scaled

def predict(model, data):
    model.eval()
    pred = torch.empty(0)
    labels = torch.empty(0)

    device = next(model.parameters()).device

    for batch_samples, batch_labels, batch_IDs in data:
        batch_samples = batch_samples.to(device)

        x = batch_samples

        pre, _, _ = model(x)

        pred = torch.cat((pred, pre.detach().cpu()), 0)
        labels = torch.cat((labels, batch_labels), dim=0)

    return pred.cpu().detach().numpy(), labels.cpu().detach().numpy()

def evaluate(pred, real):
    # 计算误差
    error = pred - real

    # 计算平均绝对误差（MAE）
    mae = np.mean(np.abs(error), axis=0)
    # 计算平均误差（Mean Error）
    mean_error = np.mean(error, axis=0)
    # 计算标准差（Standard Deviation）
    std_dev = np.std(error, axis=0)
    return mae, mean_error, std_dev