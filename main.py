import argparse
import os
from os.path import join as fullfile
import sys
import copy
import time

import numpy as np
import d2l
from hdf5storage import savemat, loadmat

from util import common_utils
from util.logger import Logger
from util.evaluation import *
from util.data_preprocessing2 import *
from PersonalizationAdapter import *
from util.training import *
from model.BCGNET import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--feature_spi', help='bool, features option', default=True)
parser.add_argument('--data_dir', default='../data/', type=str)
parser.add_argument('--signal', default='darma', type=str)
parser.add_argument('--num_channels', default=1, type=int)
parser.add_argument('--model', default='../saved_model/%s-%s-Model.pt', type=str)
parser.add_argument('--training', default=False, action='store_true')
parser.add_argument('--training_scheme', default=2, type=int)  # 1=leave-one-out, 2=adversaril-transfer-learning
parser.add_argument('--log', default='../log/%s-%s-%s.log', type=str)
parser.add_argument('--savemat', default='../result/%s-%s-%s-%s.mat', type=str)
parser.add_argument('--session', default='s3-1', type=str)
parser.add_argument('--ufrom', default=0, type=int)
parser.add_argument('--uto', default=40, type=int)
args = parser.parse_args()

common_utils.set_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# datapath = '../data/night-up_35p_FF_short3.mat'
# datapath = '../data/s3-1_update_v3.mat'
# datapath = ['../data/s2-1_v3.mat', '../data/s2-2_v3.mat', '../data/s2-3_v3.mat', '../data/s3-1_v3.mat']
datapath = ['../data/preprocessed_Data_labledpeaks_250Hz_40p.mat']
# datapath = ['../data/night-up_35p_1-76_v3.mat', '../data/night-up_35p_77-148_v3.mat', '../data/night-up_35p_149-240_v3.mat', '../data/night-up_35p_242-275_v3.mat']

signal_type = 'day'  # day
training_scheme = 'NDL'

params = {
    'finetune_strategy': 'sequential',
    'training_scheme': 'NDL',
    'batch_size': 32,
    'finetune_N': 3,
    'is_finetune': 1,
    'is_balance_dist': 0,
    'is_similarity_loss': 0,
    'is_similar_group': 0,
    'PA_model': 'RF',  # RF or MLP×
    'use_BCG': True,
    'use_PI': True,
    'use_FF': True,
    'use_LAST_BP': False,
    'grid_search': False,
}

model_choice = 'OS_BCGNET_RFPA_GridSearch0_Huber_BCG1_PI1_FF1_seed0_N10_zscore1_e20_all_fluc'
# model_choice = 'Test_LOO'

# finetune_strategy = 'sequential'  # sequential; startEnd


# is_finetune = 1

setting = f'{training_scheme}-{signal_type}-{model_choice}'

_logger = Logger(args.log % (setting, args.session, args.signal), overwrite=args.training)


def log(*args, **kwargs):
    print(*args, file=_logger, **kwargs)


def main():
    data_processor = DataProcessor(datapath, signal_type, training_scheme, params)
    data_processor.load_data()
    train_loaders, test_loaders, fine_tune_loaders = data_processor.get_loaders()

    results_set = []
    all_shap_values = []
    all_shap_features = []

    for fold_idx, (train_loader, test_loader, fine_tune_loader) in enumerate(
            zip(train_loaders, test_loaders, fine_tune_loaders)):

        log(f"Training on Fold: {fold_idx}")

        # 定义模型路径
        model_path = args.model % (setting, fold_idx)  # BP Estimator
        log(f'model_path:{model_path}')
        filepath, _ = os.path.split(model_path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # 定义模型
        model = BCGNet()
        model.to(device)

        # 训练

        if args.training:
            log('training mode')
            train_DL(log, model, model_path, train_loader, test_loader, args.lr, args.epoch, params)
        else:
            log('testing mode')
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        pred, labels = predict(model, test_loader)
        mae, mean_error, std_dev = evaluate(pred, labels)

        results = {
            'fold_idx': fold_idx,
            'pred': pred,
            'labels': labels,
        }
        results_set.append(results)

        log('[Fold:%d] ' % fold_idx)
        log('MAE: (%.4f, %.4f)' % (mae[0], mae[1]))
        log('ME : (%.4f, %.4f)' % (mean_error[0], mean_error[1]))
        log('STD: (%.4f, %.4f)' % (std_dev[0], std_dev[1]))

        if params['is_finetune']:
            if params['PA_model'] == 'MLP':
                pass  # PA_results_set = training_PA2(log, model, test_loader, fine_tune_loader, data_processor, params)
            else:
                PA_results_set, shap_values, shap_features = training_PA(log, model, test_loader, fine_tune_loader, data_processor, params)
                all_shap_values.append(shap_values)
                all_shap_features.append(shap_features)
            savemat(args.savemat % (
            setting, args.session, args.signal, str(fold_idx) + '_PA_' + params['finetune_strategy']),
                    {'PA_results': PA_results_set, 'Params': params})

    savemat(args.savemat % (setting, args.session, args.signal, fold_idx), {'results': results_set})
    shapVisual(all_shap_values, all_shap_features)

def shapVisual(all_shap_values, all_shap_features):

    feature_names = ['Gender','Age','Weight(kg)','Height(cm)',
                     'T(H,I)', 'T(H,J)', 'T(H,K)', 'T(H,L)', 'T(I,J)', 'T(I,K)', 'T(I,L)', 'T(J,K)', 'T(J,L)', 'T(K,L)',
                     'T(H,I)/T', 'T(H,J)/T', 'T(H,K)/T', 'T(I,J)/T', 'T(I,K)/T', 'T(I,L)/T', 'T(J,K)/T', 'T(J,L)/T',
                     'T(K,L)/T',
                     'A(H)', 'A(I)', 'A(J)', 'A(K)', 'A(L)',
                     '|A(H)-A(I)|', '|A(H)-A(J)|', '|A(H)-A(K)|', '|A(H)-A(L)|', '|A(I)-A(J)|', '|A(I)-A(K)|',
                     '|A(I)-A(L)|', '|A(J)-A(K)|', '|A(J)-A(L)|', '|A(K)-A(L)|',
                     'AUC(H,I)', 'AUC(H,J)', 'AUC(H,K)', 'AUC(H,L)', 'AUC(I,J)', 'AUC(I,K)', 'AUC(I,L)', 'AUC(J,K)',
                     'AUC(J,L)', 'AUC(K,L)']

    # 合并SHAP值
    all_shap_values_dbp = np.vstack([values[0] for values in all_shap_values])
    all_shap_values_sbp = np.vstack([values[1] for values in all_shap_values])

    # 合并特征数据
    all_feature_data = np.vstack(all_shap_features)

    # 计算所有特征的平均绝对SHAP值
    mean_abs_shap_values_dbp = np.abs(all_shap_values_dbp).mean(axis=0)
    mean_abs_shap_values_sbp = np.abs(all_shap_values_sbp).mean(axis=0)

    # 对每个类别获取最重要的12个特征的索引
    top_indices_dbp = np.argsort(-mean_abs_shap_values_dbp)[:12]
    top_indices_sbp = np.argsort(-mean_abs_shap_values_sbp)[:12]

    # 选择最重要的特征和对应的名称
    top_features_dbp = all_feature_data[:, top_indices_dbp]
    top_features_sbp = all_feature_data[:, top_indices_sbp]
    top_feature_names_dbp = [feature_names[i] for i in top_indices_dbp]
    top_feature_names_sbp = [feature_names[i] for i in top_indices_sbp]

    # 生成合并后的SHAP值的可视化图，只包含最重要的12个特征
    plt.figure(figsize=(13, 8))
    shap.summary_plot(all_shap_values_dbp[:, top_indices_dbp], top_features_dbp, feature_names=top_feature_names_dbp)
    plt.figure(figsize=(13, 8))
    shap.summary_plot(all_shap_values_sbp[:, top_indices_sbp], top_features_sbp, feature_names=top_feature_names_sbp)
    print('Visualized!')

if __name__ == '__main__':
    main()
