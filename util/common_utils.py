import os
import random
import pickle

import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def make_folder(filename):
    folder_path = os.path.dirname(filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_labels_from_softmax(output):
    return output.max(dim=1).indices.cpu().detach().numpy()

def auc(pos, neg):
    pos.sort(reverse=True)
    neg.sort(reverse=True)

    area = 0
    sensitivity = []
    specificity = [1]

    num_pos = 0
    for j in range(len(pos)):
        if pos[j] > neg[0]:
            num_pos += 1
        else:
            break
    sensitivity.append(float(num_pos) / len(pos))

    for i in range(len(neg)):
        threshold = neg[i]
        if i < (len(neg) - 1):
            if neg[i] == neg[i + 1]:
                continue

        num_pos = 0
        for j in range(len(pos)):
            if pos[j] >= threshold:
                num_pos += 1
            else:
                break

        specificity.append(1 - float(i + 1) / len(neg))
        sensitivity.append(float(num_pos) / len(pos))
        area += (sensitivity[-2] + sensitivity[-1]) * (specificity[-2] - specificity[-1]) / 2.0

    return area, sensitivity, specificity
