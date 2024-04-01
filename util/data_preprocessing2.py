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


class CustomDataset(Dataset):
    def __init__(self, features, labels, subjects):
        self.features = features
        self.labels = labels
        self.subjects = subjects

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.subjects[idx]


class DataProcessor:
    def __init__(self, datapath, signal_type, training_scheme, params):
        self.datapath = datapath
        self.signal_type = signal_type
        self.training_scheme = training_scheme
        self.batch_size = params['batch_size']
        self.params = params
        self.all_train_loaders = []
        self.all_test_loaders = []
        self.all_fine_tune_loaders = []

        self.fs = 250
        # self.seg_len = 1250
        self.dim_spi = 4
        self.dim_FF = 44
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.finetune_N = params['finetune_N']
        self.finetune_strategy = params['finetune_strategy']  # 新增加的微调样本选择策略
        self.window_length = 5 * self.fs  # 5秒窗口长度
        self.frame_shift = 2 * self.fs  # 3秒帧移

    def load_data(self):
        features_dict = {}
        labels_dict = {}
        all_people_group = []
        ids_dict = set()

        for path in self.datapath:
            mat = loadmat(path)
            subject_ID = mat['preprocessed_Data']['ID'][0]
            subject_NUM = subject_ID.shape[0]

            features, labels, ids, _ = self._load_subjects_data(mat, subject_NUM)
            print(f'Feature extracting: {path}')

            # 将每个文件中的用户数据添加到字典中，以用户ID为键
            for idx, user_id in enumerate(ids):
                user_id = user_id[0]
                if user_id not in features_dict:
                    features_dict[user_id] = features[idx]
                    labels_dict[user_id] = labels[idx]
                else:
                    # 如果该用户ID已存在，则将新数据追加到现有数据
                    features_dict[user_id] = torch.cat((features_dict[user_id], features[idx]), dim=0)
                    labels_dict[user_id] = torch.cat((labels_dict[user_id], labels[idx]), dim=0)

                ids_dict.add(user_id)

        # 转换字典为列表
        all_people_features = [features_dict[user_id] for user_id in sorted(ids_dict)]
        all_people_labels = [labels_dict[user_id] for user_id in sorted(ids_dict)]
        all_people_ids = sorted(list(ids_dict))

        if self.training_scheme == 'DL':
            self._process_DL(all_people_features, all_people_labels)
        elif self.training_scheme == 'NDL':
            if self.params['is_balance_dist']:
                self._process_NDL2(all_people_features, all_people_labels, all_people_ids, all_people_group)
            else:
                self._process_NDL3(all_people_features, all_people_labels, all_people_ids)
        elif self.training_scheme == 'SM':
            self._process_SM(all_people_features, all_people_labels)
        else:
            raise ValueError("Invalid training scheme")

    def load_similar_data(self, similar_group):
        mat = loadmat(self.datapath)
        subject_ID = mat['data']['ID'][0]
        subject_NUM = subject_ID.shape[0]

        all_people_features, all_people_labels, all_people_ids, all_people_group = self._load_subjects_data(mat,
                                                                                                            subject_NUM)

        # 根据similar_group,把目标用户的样本都提取出来
        similar_features = []
        similar_labels = []

        # 遍历所有用户，检查是否在 similar_group 中
        for i, user_id in enumerate(all_people_ids):
            if user_id in similar_group:
                # 如果用户在 similar_group 中，添加他们的特征和标签到列表
                # similar_features.append(all_people_features[i])
                # similar_labels.append(all_people_labels[i])

                # 相似群体随机采样10个
                if len(all_people_features[i]) > 10:
                    indices = torch.randperm(len(all_people_features[i]))[:10]
                    sampled_features = all_people_features[i][indices]
                    sampled_labels = all_people_labels[i][indices]
                else:
                    sampled_features = all_people_features[i]
                    sampled_labels = all_people_labels[i]

                similar_features.append(sampled_features)
                similar_labels.append(sampled_labels)

        # 合并所有相似用户群的特征和标签
        combined_features = torch.cat(similar_features, dim=0)
        combined_labels = torch.cat(similar_labels, dim=0)

        return combined_features, combined_labels

    def _examineFluctuation(self, sbp, dbp):
        # Downsample the data to 1Hz (from 250Hz), taking every 250th sample
        sbp_downsampled = sbp[::250]
        dbp_downsampled = dbp[::250]

        # Check for fluctuations in the downsampled data
        # Check if the difference between consecutive samples is greater than 8
        if np.any(np.abs(np.diff(sbp_downsampled)) > 8) or np.any(np.abs(np.diff(dbp_downsampled)) > 8):
            print('相邻数据变化>8')
            print(sbp_downsampled)
            return True

        # Check if the difference between the max and min values is greater than 10
        if (np.max(sbp_downsampled) - np.min(sbp_downsampled) > 10) or (
                np.max(dbp_downsampled) - np.min(dbp_downsampled) > 10):
            print('5s内最大值-最小值超过10')
            print(sbp_downsampled)
            return True

        # Check if the variance of the data is greater than 8
        if np.std(sbp_downsampled) > 5 or np.std(dbp_downsampled) > 5:
            print('整体方差过大')
            print(sbp_downsampled)
            return True

        # If none of the conditions are met, return False
        return False

    def _load_subjects_data(self, mat, subject_NUM):
        all_people_features = []
        all_people_labels = []
        all_people_ids = []  # 新增一个列表来存储所有用户的ID
        all_people_group = []
        window_length = self.window_length  # 5秒窗口长度
        frame_shift = self.frame_shift  # 3秒帧移

        # ... 加载每个用户的数据 ...

        for user_index in range(subject_NUM):
            print(f'loading {user_index}')
            # 这里的+2是留给上一时间步的收缩压和舒张压值的
            person_features = torch.empty((0, self.window_length + self.dim_spi + self.dim_FF + 2), dtype=torch.float32)
            person_labels = torch.empty((0, 2), dtype=torch.float32)

            BCG = mat['preprocessed_Data']['BCG'][0][user_index].reshape(-1)  # 可以修改第二个字段选取所需要的数据
            sbp = mat['preprocessed_Data']['SBP'][0][user_index].reshape(-1)
            dbp = mat['preprocessed_Data']['DBP'][0][user_index].reshape(-1)

            gender = mat['preprocessed_Data']['gender'][0][user_index][0][0]
            age = mat['preprocessed_Data']['age'][0][user_index][0][0]
            height = mat['preprocessed_Data']['height'][0][user_index][0][0]
            weight = mat['preprocessed_Data']['weight'][0][user_index][0][0]

            PI = torch.tensor(np.array((gender, age, weight, height)), dtype=torch.float32)

            H_locs = mat['preprocessed_Data']['H_locs'][0][user_index].astype(int).reshape(-1)
            I_locs = mat['preprocessed_Data']['I_locs'][0][user_index].astype(int).reshape(-1)
            J_locs = mat['preprocessed_Data']['J_locs'][0][user_index].astype(int).reshape(-1)
            K_locs = mat['preprocessed_Data']['K_locs'][0][user_index].astype(int).reshape(-1)
            L_locs = mat['preprocessed_Data']['L_locs'][0][user_index].astype(int).reshape(-1)

            # 初始化窗口的起始和结束点
            window_start = 0
            window_end = window_start + window_length
            last_window_DBP = 0  # np.mean(dbp[window_start:window_end])  # 上一个时间步的舒张压，初始化
            last_window_SBP = 0  # np.mean(sbp[window_start:window_end])  # 上一个时间步的收缩压，初始化
            while window_end <= len(BCG):
                # 获取当前窗口内的J_locs
                window_J_indices = np.where((J_locs >= window_start) & (J_locs < window_end))[0]

                if len(window_J_indices) == 0:
                    # 如果当前窗口内没有J_locs，跳过这个窗口
                    window_start += frame_shift
                    window_end = window_start + window_length
                    continue

                # if self._examineFluctuation(sbp[window_start:window_end],dbp[window_start:window_end]):
                #     window_start += frame_shift
                #     window_end = window_start + window_length
                #     continue

                # 根据J_locs的索引获取窗口内的所有基准点位置
                window_H_locs = H_locs[window_J_indices]
                window_I_locs = I_locs[window_J_indices]
                window_J_locs = J_locs[window_J_indices]
                window_K_locs = K_locs[window_J_indices]
                window_L_locs = L_locs[window_J_indices]

                # 计算当前窗口内的血压平均值
                window_DBP = np.mean(dbp[window_J_indices])
                window_SBP = np.mean(sbp[window_J_indices])

                #

                # 计算当前窗口内的基准点特征
                window_features = compute_features(BCG, window_H_locs, window_I_locs, window_J_locs,
                                                   window_K_locs, window_L_locs)

                # 将所有特征平均值转换为一个长向量
                time_interval = np.vstack(list(window_features['Time Interval'].values()))
                time_ratio = np.vstack(list(window_features['Time Ratio'].values()))
                extremum = np.vstack(list(window_features['Extremum'].values()))
                displacement = np.vstack(list(window_features['Displacement'].values()))
                auc = np.vstack(list(window_features['AUC'].values()))
                tmp = np.vstack((time_interval, time_ratio, extremum, displacement, auc))
                # print(np.isnan(tmp).any())
                FF = np.mean(tmp, axis=1)

                # 将BCG段、个人信息和特征合并为一个样本
                seg = BCG[window_start:window_end]
                seg_tensor = torch.tensor(seg, dtype=torch.float32)
                FF_tensor = torch.tensor(FF, dtype=torch.float32)
                LASTBP_tensor = torch.tensor([last_window_DBP, last_window_SBP], dtype=torch.float32)
                sample_features = torch.cat((seg_tensor, PI, FF_tensor, LASTBP_tensor))

                # 将当前样本添加到个人的特征和标签中间变量
                person_features = torch.cat((person_features, sample_features.reshape(1, -1)), dim=0)
                person_labels = torch.cat(
                    (person_labels, torch.tensor([[window_DBP, window_SBP]], dtype=torch.float32)), dim=0)

                # 更新窗口的起始和结束点
                window_start += frame_shift
                window_end = window_start + window_length

            # 添加样本到特征和标签列表
            all_people_features.append(person_features)
            all_people_labels.append(person_labels)

            # 添加用户ID到列表
            all_people_ids.append(mat['preprocessed_Data']['ID'][0][user_index][0])
            # all_people_group.append(mat['data']['BCG_Group'][0][user_index][0])

        return all_people_features, all_people_labels, all_people_ids, all_people_group

    def _process_DL(self, all_people_features, all_people_labels):
        # 存储五折交叉验证的结果
        fold_train_features = [torch.empty((0, all_people_features[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_test_features = [torch.empty((0, all_people_features[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_train_labels = [torch.empty((0, all_people_labels[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_test_labels = [torch.empty((0, all_people_labels[0].shape[1]), dtype=torch.float32) for _ in range(5)]

        for user_features, user_labels in zip(all_people_features, all_people_labels):
            for fold_idx, (train_idx, test_idx) in enumerate(self.kfold.split(user_features)):
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

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)

    def _process_NDL(self, all_people_features, all_people_labels, all_people_ids):
        all_features = torch.cat(all_people_features, dim=0)
        all_labels = torch.cat(all_people_labels, dim=0)
        subjects = np.concatenate(
            [[id] * len(features) for id, features in zip(all_people_ids, all_people_features)])  # 使用用户ID而不是数字索引

        unique_ids = np.array(all_people_ids)
        for train_ids, test_ids in self.kfold.split(unique_ids):
            # 使用用户ID来创建训练和测试的索引
            train_idx = np.isin(subjects, unique_ids[train_ids])
            test_idx = np.isin(subjects, unique_ids[test_ids])

            train_features = all_features[train_idx]
            train_labels = all_labels[train_idx]
            test_features = all_features[test_idx]
            test_labels = all_labels[test_idx]
            test_subjects = subjects[test_idx]

            fine_tune_features_list = []
            fine_tune_labels_list = []
            fine_tune_subjects_list = []
            remaining_test_features_list = []
            remaining_test_labels_list = []
            remaining_test_subjects_list = []

            unique_test_subjects = np.unique(test_subjects)
            for subject in unique_test_subjects:
                subject_indices = np.where(test_subjects == subject)[0]
                if len(subject_indices) >= self.finetune_N:
                    # 新增的startEnd策略
                    if self.finetune_strategy == 'startEnd':
                        half_n = self.finetune_N // 2
                        start_indices = subject_indices[:half_n]  # 取前finetune_N/2个样本
                        end_indices = subject_indices[-half_n:]  # 取后finetune_N/2个样本
                        fine_tune_indices = np.concatenate((start_indices, end_indices))
                    elif self.finetune_strategy == 'random':
                        fine_tune_indices = np.random.choice(subject_indices, self.finetune_N, replace=False)
                    elif self.finetune_strategy == 'sequential':
                        fine_tune_indices = subject_indices[:self.finetune_N]
                    else:
                        raise ValueError("Invalid finetune strategy")

                    remaining_indices = np.setdiff1d(subject_indices, fine_tune_indices)

                    fine_tune_features_list.append(test_features[fine_tune_indices])
                    fine_tune_labels_list.append(test_labels[fine_tune_indices])
                    fine_tune_subjects_list.append(test_subjects[fine_tune_indices])

                    remaining_test_features_list.append(test_features[remaining_indices])
                    remaining_test_labels_list.append(test_labels[remaining_indices])
                    remaining_test_subjects_list.append(test_subjects[remaining_indices])
                else:
                    raise ValueError(f"Subject {subject} does not have enough samples for finetuning")

            fine_tune_features = torch.cat(fine_tune_features_list, dim=0)
            fine_tune_labels = torch.cat(fine_tune_labels_list, dim=0)
            fine_tune_subjects = np.concatenate(fine_tune_subjects_list)

            remaining_test_features = torch.cat(remaining_test_features_list, dim=0)
            remaining_test_labels = torch.cat(remaining_test_labels_list, dim=0)
            remaining_test_subjects = np.concatenate(remaining_test_subjects_list)

            # 创建微调数据集的 DataLoader
            fine_tune_dataset = CustomDataset(fine_tune_features, fine_tune_labels, fine_tune_subjects)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=self.batch_size, shuffle=False)
            self.all_fine_tune_loaders.append(fine_tune_loader)

            # 创建训练和测试数据集的 DataLoader
            train_dataset = CustomDataset(train_features, train_labels, subjects[train_idx])
            test_dataset = CustomDataset(remaining_test_features, remaining_test_labels, remaining_test_subjects)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)

    def _process_NDL2(self, all_people_features, all_people_labels, all_people_ids, all_people_groups):
        # 初始化用于存储训练、测试和微调数据的结构
        fold_train_features = [[] for _ in range(5)]
        fold_test_features = [[] for _ in range(5)]
        fold_train_labels = [[] for _ in range(5)]
        fold_test_labels = [[] for _ in range(5)]
        fold_fine_tune_features = [[] for _ in range(5)]
        fold_fine_tune_labels = [[] for _ in range(5)]
        fold_fine_tune_subjects = [[] for _ in range(5)]
        fold_test_subjects = [[] for _ in range(5)]
        fold_train_subjects = [[] for _ in range(5)]
        # 分层抽样
        for group in [0, 1, 2, 3]:
            group_indices = [i for i, g in enumerate(all_people_groups) if g == group]
            group_features = [all_people_features[i] for i in group_indices]
            group_labels = [all_people_labels[i] for i in group_indices]
            group_subjects = [all_people_ids[i] for i in group_indices]

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(group_features)):
                # 训练集
                fold_train_features[fold_idx].extend([group_features[i] for i in train_idx])
                fold_train_labels[fold_idx].extend([group_labels[i] for i in train_idx])
                fold_train_subjects[fold_idx].extend(
                    [np.array([group_subjects[i]] * len(group_features[i])) for i in train_idx])

                # 测试集和微调集
                test_and_fine_tune_features = [group_features[i] for i in test_idx]
                test_and_fine_tune_labels = [group_labels[i] for i in test_idx]
                test_and_fine_tune_subjects = [group_subjects[i] for i in test_idx]
                test_and_fine_tune_subjects = np.array(test_and_fine_tune_subjects)

                fine_tune_indices = []
                remaining_test_indices = []
                for i, subject_index in enumerate(test_and_fine_tune_subjects):
                    N = len(test_and_fine_tune_features[i])
                    if N >= self.finetune_N:
                        if self.finetune_strategy == 'random':
                            selected_indices = random.sample(range(N), self.finetune_N)
                        elif self.finetune_strategy == 'sequential':
                            selected_indices = list(range(self.finetune_N))
                        elif self.finetune_strategy == 'startEnd':
                            half_n = self.finetune_N // 2
                            selected_indices = list(range(half_n)) + list(range(N - half_n, len(N)))
                        else:
                            raise ValueError("Invalid finetune strategy")

                        fine_tune_features = [test_and_fine_tune_features[i][index] for index in selected_indices]
                        fine_tune_labels = [test_and_fine_tune_labels[i][index] for index in selected_indices]
                        fine_tune_features = torch.vstack(fine_tune_features)
                        fine_tune_labels = torch.vstack(fine_tune_labels)
                        fine_tune_subjects = np.array([subject_index] * self.finetune_N)

                        fold_fine_tune_features[fold_idx].extend([fine_tune_features])
                        fold_fine_tune_labels[fold_idx].extend([fine_tune_labels])
                        fold_fine_tune_subjects[fold_idx].extend([fine_tune_subjects])

                        remaining_indices = [i for i in range(N) if i not in selected_indices]
                        remaining_features = [test_and_fine_tune_features[i][index] for index in remaining_indices]
                        remaining_labels = [test_and_fine_tune_labels[i][index] for index in remaining_indices]
                        remaining_features = torch.vstack(remaining_features)
                        remaining_labels = torch.vstack(remaining_labels)
                        remaining_subjects = np.array([subject_index] * (N - self.finetune_N))

                        fold_test_features[fold_idx].extend([remaining_features])
                        fold_test_labels[fold_idx].extend([remaining_labels])
                        fold_test_subjects[fold_idx].extend([remaining_subjects])
                    else:
                        raise ValueError(f"User with index {subject_index} does not have enough samples for finetuning")

                # fold_fine_tune_features[fold_idx].extend([test_and_fine_tune_features[i] for i in fine_tune_indices])
                # fold_fine_tune_labels[fold_idx].extend([test_and_fine_tune_labels[i] for i in fine_tune_indices])
                # fold_test_features[fold_idx].extend([test_and_fine_tune_features[i] for i in remaining_test_indices])
                # fold_test_labels[fold_idx].extend([test_and_fine_tune_labels[i] for i in remaining_test_indices])

        # 创建 DataLoader
        self.all_train_loaders = []
        self.all_test_loaders = []
        self.all_fine_tune_loaders = []
        for i in range(5):
            train_dataset = CustomDataset(torch.cat(fold_train_features[i]), torch.cat(fold_train_labels[i]),
                                          np.concatenate(fold_train_subjects[i]))
            test_dataset = CustomDataset(torch.cat(fold_test_features[i]), torch.cat(fold_test_labels[i]),
                                         np.concatenate(fold_test_subjects[i]))
            fine_tune_dataset = CustomDataset(torch.cat(fold_fine_tune_features[i]),
                                              torch.cat(fold_fine_tune_labels[i]),
                                              np.concatenate(fold_fine_tune_subjects[i]))

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)
            self.all_fine_tune_loaders.append(fine_tune_loader)

    def _process_NDL3(self, all_people_features, all_people_labels, all_people_ids):
        all_features = torch.cat(all_people_features, dim=0)
        all_labels = torch.cat(all_people_labels, dim=0)
        subjects = np.concatenate(
            [[id] * len(features) for id, features in zip(all_people_ids, all_people_features)])  # 使用用户ID而不是数字索引

        unique_ids = np.array(all_people_ids)
        for loo_id in unique_ids:
            # 留一个样本作为测试集
            test_idx = np.where(subjects == loo_id)[0]
            train_idx = np.where(subjects != loo_id)[0]

            train_features = all_features[train_idx]
            train_labels = all_labels[train_idx]
            test_features = all_features[test_idx]
            test_labels = all_labels[test_idx]
            test_subjects = subjects[test_idx]

            fine_tune_features_list = []
            fine_tune_labels_list = []
            fine_tune_subjects_list = []
            remaining_test_features_list = []
            remaining_test_labels_list = []
            remaining_test_subjects_list = []

            unique_test_subjects = np.unique(test_subjects)
            for subject in unique_test_subjects:
                subject_indices = np.where(test_subjects == subject)[0]
                if len(subject_indices) >= self.finetune_N:
                    # 新增的startEnd策略
                    if self.finetune_strategy == 'startEnd':
                        half_n = self.finetune_N // 2
                        start_indices = subject_indices[:half_n]  # 取前finetune_N/2个样本
                        end_indices = subject_indices[-half_n:]  # 取后finetune_N/2个样本
                        fine_tune_indices = np.concatenate((start_indices, end_indices))
                    elif self.finetune_strategy == 'random':
                        fine_tune_indices = np.random.choice(subject_indices, self.finetune_N, replace=False)
                    elif self.finetune_strategy == 'sequential':
                        fine_tune_indices = subject_indices[:self.finetune_N]
                    else:
                        raise ValueError("Invalid finetune strategy")

                    remaining_indices = np.setdiff1d(subject_indices, fine_tune_indices)

                    fine_tune_features_list.append(test_features[fine_tune_indices])
                    fine_tune_labels_list.append(test_labels[fine_tune_indices])
                    fine_tune_subjects_list.append(test_subjects[fine_tune_indices])

                    remaining_test_features_list.append(test_features[remaining_indices])
                    remaining_test_labels_list.append(test_labels[remaining_indices])
                    remaining_test_subjects_list.append(test_subjects[remaining_indices])
                else:
                    raise ValueError(f"Subject {subject} does not have enough samples for finetuning")

            fine_tune_features = torch.cat(fine_tune_features_list, dim=0)
            fine_tune_labels = torch.cat(fine_tune_labels_list, dim=0)
            fine_tune_subjects = np.concatenate(fine_tune_subjects_list)

            remaining_test_features = torch.cat(remaining_test_features_list, dim=0)
            remaining_test_labels = torch.cat(remaining_test_labels_list, dim=0)
            remaining_test_subjects = np.concatenate(remaining_test_subjects_list)

            # 创建微调数据集的 DataLoader
            fine_tune_dataset = CustomDataset(fine_tune_features, fine_tune_labels, fine_tune_subjects)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=self.batch_size, shuffle=False)
            self.all_fine_tune_loaders.append(fine_tune_loader)

            # 创建训练和测试数据集的 DataLoader
            train_dataset = CustomDataset(train_features, train_labels, subjects[train_idx])
            test_dataset = CustomDataset(remaining_test_features, remaining_test_labels, remaining_test_subjects)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)

    def _process_SM(self, all_people_features, all_people_labels):
        # ... 处理SM训练方案 ...
        pass

    def get_loaders(self):
        return self.all_train_loaders, self.all_test_loaders, self.all_fine_tune_loaders


if __name__ == '__main__':
    datapath = '../../data/s3-1_update_v3.mat'
    # datapath = '../data/night-up_35p_FF_short3.mat'
    signal_type = 'day'
    training_scheme = 'NDL'
    batch_size = 64
    finetune_N = 10
    finetune_strategy = 'sequential'  # random

    data_processor = DataProcessor(datapath, signal_type, training_scheme, batch_size, finetune_N, finetune_strategy)
    data_processor.load_data()
    train_loaders, test_loaders, fine_tune_loaders = data_processor.get_loaders()
    print("Done")
