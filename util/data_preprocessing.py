import numpy as np
import torch
from hdf5storage import loadmat, savemat
from scipy.stats import zscore
import random
import math
import pandas as pd
from sklearn import preprocessing
import os
import re
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy import signal
import time
from util.fiducial_feature_extraction import *
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold


def scale_F_BCG_train(F_BCG, current_dbp, current_sbp):

    if len(F_BCG) == 0:
        return F_BCG  # Return the empty array as is

    F_BCG_scaled = ((F_BCG - np.min(F_BCG)) / (np.max(F_BCG) - np.min(F_BCG))) * (
                current_sbp - current_dbp) + current_dbp
    return F_BCG_scaled

def scale_F_BCG(F_BCG_original, current_dbp, current_sbp, ref_BCG):
    if len(F_BCG_original) == 0:
        return F_BCG_original  # Return the empty array as is

    min_val = torch.min(ref_BCG)
    max_val = torch.max(ref_BCG)

    F_BCG_scaled = ((F_BCG_original - min_val) / (max_val - min_val + 1e-8)) * (current_sbp - current_dbp) + current_dbp

    return F_BCG_scaled


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def loaddata(datapath, signal_type, training_scheme, batch_size=64):
    """
    filename: 定义数据的路径
    singla_type: "day" or "night"
    training_scheme: "DL" for data Leakage，训练一个大模型，但里面包含每个人的数据; "NDL" for no data leakage, 训练测试验证集是不同的人
                    "DA" for domain adaptation, 'SM'for single model for each one
    """

    fs = 250
    seg_len = 200
    dim_spi = 4
    dim_FF = 44

    mat = loadmat(datapath)
    subject_ID = mat['data']['ID'][0]
    subject_NUM = subject_ID.shape[0]

    all_people_features = []
    all_people_labels = []
    for user_index in range(subject_NUM):
        print(f'loading {user_index}')

        features = torch.empty((0, seg_len + dim_spi + dim_FF), dtype=torch.float32)
        labels = torch.empty((0, 2), dtype=torch.float32)

        BCG = mat['data']['darma_zscore_after_nk_filter'][0][user_index].reshape(-1)  # 可以修改第二个字段选取所需要的数据
        sbp = mat['data']['sbp'][0][user_index].reshape(-1)
        dbp = mat['data']['dbp'][0][user_index].reshape(-1)

        gender = mat['data']['gender'][0][user_index][0][0]
        age = mat['data']['age'][0][user_index][0][0]
        height = mat['data']['height'][0][user_index][0][0]
        weight = mat['data']['weight'][0][user_index][0][0]

        PI = torch.tensor(np.array((gender, age, weight, height)), dtype=torch.float32)

        H_locs = mat['data']['H_locs'][0][user_index].reshape(-1)
        I_locs = mat['data']['I_locs'][0][user_index].reshape(-1)
        J_locs = mat['data']['J_locs'][0][user_index].reshape(-1)
        K_locs = mat['data']['K_locs'][0][user_index].reshape(-1)
        L_locs = mat['data']['L_locs'][0][user_index].reshape(-1)

        fiducial_features = compute_features(BCG, H_locs, I_locs, J_locs, K_locs, L_locs)

        time_interval = np.vstack(list(fiducial_features['Time Interval'].values()))
        time_ratio = np.vstack(list(fiducial_features['Time Ratio'].values()))
        extremum = np.vstack(list(fiducial_features['Extremum'].values()))
        displacement = np.vstack(list(fiducial_features['Displacement'].values()))
        auc = np.vstack(list(fiducial_features['AUC'].values()))

        # 获取J_locs前后的数据 作为1个segment
        s = (J_locs - 80).astype(int)
        e = s + seg_len
        for seg_index in range(len(J_locs)):
            seg = BCG[s[seg_index]:e[seg_index]]
            # seg = scale_F_BCG_train(seg, dbp[J_locs[seg_index]], sbp[J_locs[seg_index]])
            seg = torch.tensor(seg, dtype=torch.float32)
            if seg.shape[0] != seg_len:
                print(seg.shape)
                print('seg shape error')
                continue
            # plt.plot(seg)
            # plt.show()

            DBP = torch.tensor(dbp[J_locs[seg_index]], dtype=torch.float32)
            SBP = torch.tensor(sbp[J_locs[seg_index]], dtype=torch.float32)
            FF = np.hstack((time_interval[:, seg_index], time_ratio[:, seg_index], extremum[:, seg_index],
                            displacement[:, seg_index], auc[:, seg_index]))
            FF = torch.tensor(FF, dtype=torch.float32)

            # 归一化
            # seg = F.normalize(seg, dim=0)

            # PI = F.normalize(PI, dim=0)
            # FF = F.normalize(FF, dim=0)

            features = torch.cat((features, torch.hstack((seg, PI, FF)).reshape(1, -1)), dim=0)
            labels = torch.cat((labels, torch.hstack((DBP, SBP)).reshape(1, -1)), dim=0)

        all_people_features.append(features)
        all_people_labels.append(labels)

    all_train_loaders = []
    all_test_loaders = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    if training_scheme == 'DL':
        # 存储五折交叉验证的结果
        fold_train_features = [torch.empty((0, all_people_features[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_test_features = [torch.empty((0, all_people_features[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_train_labels = [torch.empty((0, all_people_labels[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_test_labels = [torch.empty((0, all_people_labels[0].shape[1]), dtype=torch.float32) for _ in range(5)]

        for user_features, user_labels in zip(all_people_features, all_people_labels):
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(user_features)):
                train_features = user_features[train_idx]
                train_labels = user_labels[train_idx]
                test_features = user_features[test_idx]
                test_labels = user_labels[test_idx]

                fold_train_features[fold_idx] = torch.cat((fold_train_features[fold_idx], train_features), dim=0)
                fold_train_labels[fold_idx] = torch.cat((fold_train_labels[fold_idx], train_labels), dim=0)
                fold_test_features[fold_idx] = torch.cat((fold_test_features[fold_idx], test_features), dim=0)
                fold_test_labels[fold_idx] = torch.cat((fold_test_labels[fold_idx], test_labels), dim=0)

        # 创建整体五折训练和测试Dataloader
        for train_features, train_labels, test_features, test_labels in zip(fold_train_features, fold_train_labels,
                                                                            fold_test_features, fold_test_labels):
            train_dataset = CustomDataset(train_features, train_labels)
            test_dataset = CustomDataset(test_features, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            all_train_loaders.append(train_loader)
            all_test_loaders.append(test_loader)

    elif training_scheme == 'NDL':
        all_features = torch.cat(all_people_features, dim=0)
        all_labels = torch.cat(all_people_labels, dim=0)
        subjects = np.concatenate([[i] * all_people_features[i].shape[0] for i in range(subject_NUM)])

        for train_subject_idx, test_subject_idx in kfold.split(np.unique(subjects)):
            train_idx = np.isin(subjects, train_subject_idx)
            test_idx = np.isin(subjects, test_subject_idx)

            # # For each subject in the test set, use their first sample as ref_BCG and scale their features
            # for sub in test_subject_idx:
            #     ref_BCG = all_features[subjects == sub][0, :]  # The first sample for this subject
            #     ref_label = all_labels[subjects == sub][0,
            #                 :]  # The label (current_dbp, current_sbp) for the first sample
            #
            #     current_dbp, current_sbp = ref_label[0], ref_label[1]
            #
            #     # Scale the features for this subject using ref_BCG
            #     mask = (subjects == sub)
            #     all_features[mask] = scale_F_BCG(all_features[mask], current_dbp, current_sbp, ref_BCG)

            train_features = all_features[train_idx]
            train_labels = all_labels[train_idx]
            test_features = all_features[test_idx]
            test_labels = all_labels[test_idx]

            train_dataset = CustomDataset(train_features, train_labels)
            test_dataset = CustomDataset(test_features, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            all_train_loaders.append(train_loader)
            all_test_loaders.append(test_loader)
    elif training_scheme == 'SM':
        all_train_loaders = []
        all_test_loaders = []

        for user_features, user_labels in zip(all_people_features, all_people_labels):
            # 划分每个人的数据为训练集和测试集
            train_features, test_features, train_labels, test_labels = train_test_split(user_features, user_labels,
                                                                                        test_size=0.40, random_state=42)

            # 创建数据集
            train_dataset = CustomDataset(train_features, train_labels)
            test_dataset = CustomDataset(test_features, test_labels)

            # 创建加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            all_train_loaders.append(train_loader)
            all_test_loaders.append(test_loader)

    return all_train_loaders, all_test_loaders

    if signal_type == 'day':
        print('processing day-time data')
    elif signal_type == 'night':
        print('processing night-time data')


def mean_regressor():
    datapath = '../../data/night-up_35p.mat'
    mat = loadmat(datapath)
    subject_ID = mat['data']['ID'][0]
    subject_NUM = subject_ID.shape[0]

    pred = np.empty(0)
    label = np.empty(0)
    results_set = []
    for user_index in range(subject_NUM):
        print(f'loading {user_index}')

        sbp = mat['data']['sbp'][0][user_index].reshape(-1)
        dbp = mat['data']['dbp'][0][user_index].reshape(-1)

        sbp_mean_pred = np.full_like(sbp, np.mean(sbp))
        dbp_mean_pred = np.full_like(dbp, np.mean(dbp))

        pred = np.vstack((dbp_mean_pred, sbp_mean_pred))
        label = np.vstack((dbp, sbp))

        results = {
            'uid': user_index,
            'pred': pred,
            'labels': label,
        }
        results_set.append(results)
    savemat('../../result/mean_regressor_day.mat', {'results': results_set})

# mean_regressor()
