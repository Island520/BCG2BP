import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from model.BCGNET import *
from util.data_preprocessing2 import *
from util.evaluation import *
import shap

def get_similar_group_for_finetue(log, current_id, data_path):
    datapath = '../data/s3-1_update_v3.mat'
    u_data_dict = map_data_to_dict(data_path)
    current_user_dict = u_data_dict[current_id]

    current_gender = current_user_dict['gender']
    current_age = current_user_dict['age']
    current_hypertension = current_user_dict['BCG_Group']
    similar_group = []

    for user_id, user_data in u_data_dict.items():
        if user_id != current_id:
            if (user_data['gender'] == current_gender and
                    ((abs(user_data['age'] - current_age) <= 5) or (current_age >= 50 and user_data['age'] >= 50))):
                if (user_data['BCG_Group'] == 0 and current_hypertension == 0) or (
                        user_data['BCG_Group'] > 0 and current_hypertension > 0):
                    log(current_user_dict)
                    log(user_id, user_data)
                    similar_group.append(user_id)

    return similar_group


def map_data_to_dict(data_path):
    """
    将MAT文件中的数据映射到字典中。

    :param data_path: MAT文件的路径
    :return: 映射后的字典
    """
    mat = loadmat(data_path)
    subject_ID = mat['data']['ID'][0]
    subject_Num = subject_ID.shape[0]

    user_data_dict = {}

    for user_index in range(subject_Num):
        user_id = subject_ID[user_index][0]

        # 提取用户的相关数据
        user_data = {
            'gender': mat['data']['gender'][0][user_index][0][0],
            'age': int(mat['data']['age'][0][user_index][0][0]),
            'height': mat['data']['height'][0][user_index][0][0],
            'weight': mat['data']['weight'][0][user_index][0][0],
            'BCG_Group': mat['data']['BCG_Group'][0][user_index][0][0],
            # 可以继续添加其他需要的数据字段
        }

        # 将用户数据添加到字典中
        user_data_dict[user_id] = user_data

    return user_data_dict


def extract_features_from_model(model, loader):
    model.eval()  # Set the model to evaluation mode
    rawBCG_feature = []
    mid_features = []
    covar_features = []
    labels = []
    last_BP_set = []
    device = next(model.parameters()).device

    with torch.no_grad():  # No need to track gradients for this operation
        for batch_features, batch_labels, _ in loader:
            batch_features = batch_features.to(device)  # Assuming you have a device variable for cuda or cpu
            batch_labels = batch_labels.to(device)
            raw_BCG = batch_features[:, : -model.dim_PI - model.dim_FF - 2]
            last_BP = batch_features[:, -2:]
            covar = batch_features[:, -model.dim_PI - model.dim_FF - 2:- 2]
            output, mid_feature, _ = model(batch_features[:, :-2])

            mid_features.append(mid_feature)
            covar_features.append(covar)
            labels.append(batch_labels)
            rawBCG_feature.append(raw_BCG)
            last_BP_set.append(last_BP)

    return torch.cat(rawBCG_feature, dim=0), torch.cat(mid_features, dim=0), torch.cat(covar_features,
                                                                                       dim=0), torch.cat(labels,
                                                                                                         dim=0), torch.cat(
        last_BP_set, dim=0)


def training_PA2(log, model, test_loader, fine_tune_loader, data_processor, params):
    device = next(model.parameters()).device
    # 网格搜索参数
    param_grid = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 2, 3, 4],
        'gamma': [0, 0.1, 0.2, 0.4],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
        'n_estimators': [25, 30, 35, 40, 45],
        'learning_rate': [0.1, 0.2, 0.3]
    }
    print(param_grid)

    PA_results_set = []
    # 主循环
    for id in np.unique(test_loader.dataset.subjects):

        log(f'当前微调: {id}')

        # 从fine_tune_loader中提取特征和标签
        _, fine_tune_mid_features, fine_tune_covar_features, fine_tune_labels, last_BP = extract_features_from_model(
            model,
            fine_tune_loader)

        # 找到对应ID的微调数据
        fine_tune_mask = np.array([id_item == id for id_item in fine_tune_loader.dataset.subjects])
        fine_tune_indices = np.where(fine_tune_mask)[0]

        if len(fine_tune_indices) > 0:
            # 获取对应ID的微调特征和标签
            specific_fine_tune_mid_features = fine_tune_mid_features[fine_tune_indices]
            specific_fine_tune_covar_features = fine_tune_covar_features[fine_tune_indices]
            specific_fine_tune_labels = fine_tune_labels[fine_tune_indices]
            last_BP = last_BP[fine_tune_indices]

            # 拼接中间特征和协变量特征；last BP是临时添加的
            fine_tune_features = torch.cat(
                (specific_fine_tune_mid_features, specific_fine_tune_covar_features, last_BP),
                dim=1)
            fine_tune_labels = specific_fine_tune_labels

            # 找同样类型的人群
            if params['is_similar_group']:
                similar_group = get_similar_group_for_finetue(log, id, data_path='../data/s3-1_update_v3.mat')
            else:
                similar_group = []

            if len(similar_group) > 0:
                similar_features, similar_labels = data_processor.load_similar_data(similar_group)
                similar_features = similar_features.to(device)
                similar_labels = similar_labels.to(device)

                fine_tune_features = torch.vstack((similar_features[:, :-2], fine_tune_features))
                fine_tune_labels = torch.vstack((similar_labels, fine_tune_labels))
                last_BP = torch.vstack((similar_features[:, -2:], last_BP))

            # fine_tune_features = fine_tune_features.detach().cpu().numpy()

            # 输出微调血压值
            # log(f'{fine_tune_labels}')

            selected_features = []
            DL_feature_dim = 79
            if params['use_BCG']:
                selected_features.append(fine_tune_features[:, :DL_feature_dim])
            if params['use_PI']:
                selected_features.append(fine_tune_features[:, DL_feature_dim:-44 - 2])
            if params['use_FF']:
                selected_features.append(fine_tune_features[:, DL_feature_dim + 4: -2])
            if params['use_LAST_BP']:
                selected_features.append(fine_tune_features[:, -2:])

            # 将选择的特征拼接成最终的fine_tune_features_selected
            fine_tune_features_selected = torch.cat(selected_features, dim=1).detach().cpu().numpy()
            fine_tune_labels = fine_tune_labels.detach().cpu().numpy()

            # 初始化随机森林回归器并训练
            # regressor = RandomForestRegressor(**regressor_config) #Ridge(**ridge_config) #
            # regressor.fit(fine_tune_features_selected, fine_tune_labels)
            # xgb_regressor.fit(fine_tune_features_selected, fine_tune_labels)
            # 初始化XGBoost回归器
            xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')

            # 使用网格搜索训练XGBoost模型
            grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, scoring='neg_mean_squared_error',
                                       cv=3, verbose=1, n_jobs=-1)

            grid_search.fit(fine_tune_features_selected, fine_tune_labels)

            # 使用最佳参数模型
            best_model = grid_search.best_estimator_

            log("Best parameters found: ", grid_search.best_params_)
            log("Best score found: ", grid_search.best_score_)

            # 从test_loader中提取特征和标签
            _, test_mid_features, test_covar_features, test_labels, last_BP = extract_features_from_model(model,
                                                                                                          test_loader)

            # 找到对应ID的测试数据
            test_mask = np.array([id_item == id for id_item in test_loader.dataset.subjects])
            test_indices = np.where(test_mask)[0]
            log(f'找到{len(test_indices)}个测试样本')
            if len(test_indices) > 0:
                specific_test_mid_features = test_mid_features[test_indices]
                specific_test_covar_features = test_covar_features[test_indices]
                specific_test_labels = test_labels[test_indices]
                last_BP = last_BP[test_indices]

                test_selected_features = []
                if params['use_BCG']:
                    test_selected_features.append(specific_test_mid_features[:, :DL_feature_dim])
                if params['use_PI']:
                    test_selected_features.append(specific_test_covar_features[:, :4])  # 假设 PI 特征在协变量特征中的前4维
                if params['use_FF']:
                    test_selected_features.append(specific_test_covar_features[:, 4:48])  # 假设 FF 特征在协变量特征中紧随 PI 特征
                if params['use_LAST_BP']:
                    test_selected_features.append(last_BP)

                # 将选择的特征拼接成最终的test_features_selected
                test_features_selected = torch.cat(test_selected_features, dim=1).cpu().numpy()
                test_labels = specific_test_labels.cpu().numpy()

                # 使用随机森林模型进行预测
                # predictions = regressor.predict(test_features_selected)
                # predictions = xgb_regressor.predict(test_features_selected)
                predictions = best_model.predict(test_features_selected)

                # 评估模型性能
                mae, mean_error, std_dev = evaluate(predictions, test_labels)

                results = {
                    'ID': id,
                    'pred': predictions,
                    'labels': test_labels,
                }
                PA_results_set.append(results)

                log('[%s] ' % id)
                log('MAE: (%.4f, %.4f)' % (mae[0], mae[1]))
                log('ME : (%.4f, %.4f)' % (mean_error[0], mean_error[1]))
                log('STD: (%.4f, %.4f)' % (std_dev[0], std_dev[1]))

    return PA_results_set


def training_PA(log, model, test_loader, fine_tune_loader, data_processor, params):
    device = next(model.parameters()).device
    regressor_config = {
        'n_estimators': 10,
        'max_depth': 4,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }

    regressor_config = {
        'n_estimators': 20,
        'max_depth': 8,
        'min_samples_split': 2,
        'min_samples_leaf': 2
    }

    # 参数网格
    param_grid = {
        'n_estimators': [10, 20, 30, 40, 50, 60],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    print(param_grid)
    PA_results_set = []
    # 主循环
    for id in np.unique(test_loader.dataset.subjects):

        log(f'当前微调: {id}')

        # 从fine_tune_loader中提取特征和标签
        _, fine_tune_mid_features, fine_tune_covar_features, fine_tune_labels, last_BP = extract_features_from_model(
            model,
            fine_tune_loader)

        # 找到对应ID的微调数据
        fine_tune_mask = np.array([id_item == id for id_item in fine_tune_loader.dataset.subjects])
        fine_tune_indices = np.where(fine_tune_mask)[0]

        if len(fine_tune_indices) > 0:
            # 获取对应ID的微调特征和标签
            specific_fine_tune_mid_features = fine_tune_mid_features[fine_tune_indices]
            specific_fine_tune_covar_features = fine_tune_covar_features[fine_tune_indices]
            specific_fine_tune_labels = fine_tune_labels[fine_tune_indices]
            last_BP = last_BP[fine_tune_indices]

            # 拼接中间特征和协变量特征；last BP是临时添加的
            fine_tune_features = torch.cat(
                (specific_fine_tune_mid_features, specific_fine_tune_covar_features, last_BP),
                dim=1)
            fine_tune_labels = specific_fine_tune_labels

            # 找同样类型的人群
            if params['is_similar_group']:
                similar_group = get_similar_group_for_finetue(log, id, data_path='../data/s3-1_update_v3.mat')
            else:
                similar_group = []

            if len(similar_group) > 0:
                similar_features, similar_labels = data_processor.load_similar_data(similar_group)
                similar_features = similar_features.to(device)
                similar_labels = similar_labels.to(device)

                fine_tune_features = torch.vstack((similar_features[:, :-2], fine_tune_features))
                fine_tune_labels = torch.vstack((similar_labels, fine_tune_labels))
                last_BP = torch.vstack((similar_features[:, -2:], last_BP))

            # fine_tune_features = fine_tune_features.detach().cpu().numpy()

            # 输出微调血压值
            log(f'{fine_tune_labels}')

            selected_features = []
            DL_feature_dim = 79
            if params['use_BCG']:
                selected_features.append(fine_tune_features[:, :DL_feature_dim])
            if params['use_PI']:
                selected_features.append(fine_tune_features[:, DL_feature_dim:-44 - 2])
            if params['use_FF']:
                selected_features.append(fine_tune_features[:, DL_feature_dim + 4: -2])
            if params['use_LAST_BP']:
                selected_features.append(fine_tune_features[:, -2:])

            # 将选择的特征拼接成最终的fine_tune_features_selected
            fine_tune_features_selected = torch.cat(selected_features, dim=1).detach().cpu().numpy()

            fine_tune_labels = fine_tune_labels.detach().cpu().numpy()
            # 初始化随机森林回归器并训练
            if params['grid_search']:
                # 使用网格搜索找到最优参数
                regressor = RandomForestRegressor()
                grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error',
                                           cv=3, verbose=1, n_jobs=-1)

                grid_search.fit(fine_tune_features_selected, fine_tune_labels)
                best_model = grid_search.best_estimator_

                log("Best parameters found: ", grid_search.best_params_)
                log("Best score found: ", grid_search.best_score_)
            else:
                regressor = RandomForestRegressor(**regressor_config)  # Ridge(**ridge_config) #
                regressor.fit(fine_tune_features_selected, fine_tune_labels)

            # Shap visualization
            is_shap = 1
            if is_shap:
                explainer = shap.TreeExplainer(regressor)

                # 计算SHAP值
                shap_values = explainer.shap_values(fine_tune_features_selected)


            # 从test_loader中提取特征和标签
            _, test_mid_features, test_covar_features, test_labels, last_BP = extract_features_from_model(model,
                                                                                                          test_loader)

            # 找到对应ID的测试数据
            test_mask = np.array([id_item == id for id_item in test_loader.dataset.subjects])
            test_indices = np.where(test_mask)[0]
            log(f'找到{len(test_indices)}个测试样本')
            if len(test_indices) > 0:
                specific_test_mid_features = test_mid_features[test_indices]
                specific_test_covar_features = test_covar_features[test_indices]
                specific_test_labels = test_labels[test_indices]
                last_BP = last_BP[test_indices]

                test_selected_features = []
                if params['use_BCG']:
                    test_selected_features.append(specific_test_mid_features[:, :DL_feature_dim])
                if params['use_PI']:
                    test_selected_features.append(specific_test_covar_features[:, :4])  # 假设 PI 特征在协变量特征中的前4维
                if params['use_FF']:
                    test_selected_features.append(specific_test_covar_features[:, 4:48])  # 假设 FF 特征在协变量特征中紧随 PI 特征
                if params['use_LAST_BP']:
                    test_selected_features.append(last_BP)

                # 将选择的特征拼接成最终的test_features_selected
                test_features_selected = torch.cat(test_selected_features, dim=1).cpu().numpy()
                test_labels = specific_test_labels.cpu().numpy()

                # 使用随机森林模型进行预测
                if params['grid_search']:
                    predictions = best_model.predict(test_features_selected)
                else:
                    predictions = regressor.predict(test_features_selected)

                # 评估模型性能
                mae, mean_error, std_dev = evaluate(predictions, test_labels)

                results = {
                    'ID': id,
                    'pred': predictions,
                    'labels': test_labels,
                }
                PA_results_set.append(results)

                log('[%s] ' % id)
                log('MAE: (%.4f, %.4f)' % (mae[0], mae[1]))
                log('ME : (%.4f, %.4f)' % (mean_error[0], mean_error[1]))
                log('STD: (%.4f, %.4f)' % (std_dev[0], std_dev[1]))

    return PA_results_set, shap_values, fine_tune_features_selected

