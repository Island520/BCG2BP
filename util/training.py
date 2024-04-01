import time
import copy
import math

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class SimilarityLoss(nn.Module):  # 注意继承 nn.Module
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    # 清泳实现版
    def forward(self, out, labels):
        rho = torch.cosine_similarity(out - out.mean(), labels - labels.mean(), dim=-1).mean()
        loss = 1 - rho
        return loss

def scale_F_BCG(F_BCG, current_dbp, current_sbp):
    if len(F_BCG) == 0:
        return F_BCG  # Return the empty array as is

    min_val, _ = torch.min(F_BCG, dim=1, keepdim=True)
    max_val, _ = torch.max(F_BCG, dim=1, keepdim=True)

    F_BCG_scaled = ((F_BCG - min_val) / (max_val - min_val + 1e-8)) * (
                current_sbp.unsqueeze(-1) - current_dbp.unsqueeze(-1)) + current_dbp.unsqueeze(-1)

    return F_BCG_scaled

def train_DL(log, model, model_path, train_dataloader, validation_dataloader, lr, epoch, params):
    best_model = copy.deepcopy(model)

    device = next(model.parameters()).device
    dim_PI = model.dim_PI
    dim_FF = model.dim_FF

    loss_regression = nn.HuberLoss(reduction='mean', delta=5) #nn.MSELoss()
    loss_reconstruction = nn.MSELoss()
    # loss_similarity1 = SimilarityLoss()#nn.CosineSimilarity()
    # loss_similarity2 = SimilarityLoss()#nn.CosineSimilarity()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4, weight_decay=1e-5)

    best_val_loss = 1000

    for e in range(epoch):
        st = time.time()
        loss_sum, loss_val_sum, train_steps, val_steps = 0, 0, 0, 0
        loss_s1, loss_s2, loss_val_s1, loss_val_s2 = 0, 0, 0, 0

        model.train()
        for step, (features, labels, IDs) in enumerate(train_dataloader):
            train_steps = train_steps + 1

            optimizer.zero_grad()

            features = features.to(device)
            labels = labels.to(device)

            # raw_bcg = features[:, 0:-dim_PI - dim_FF - 2]
            # raw_bcg = scale_F_BCG(raw_bcg, labels[:,0], labels[:,1])
            # covar = features[:, -dim_PI - dim_FF: -2]
            x = features
            last_BP = features[:, -2:] # DBP, SBP

            regression_out, _, recon_bcg = model(x)
            regression_loss = loss_regression(regression_out, labels)
            reconstruction_loss = loss_reconstruction(recon_bcg.squeeze(1), x[:,0:1250])
            # print(f'Training: Last_BP: {last_BP.shape}, Labels: {labels.shape}')


            # 更新模型
            if params['is_similarity_loss']:
                pass # loss_total = regression_loss + si_loss1 + si_loss2
            else:
                loss_total = regression_loss + reconstruction_loss
            loss_total.backward()
            optimizer.step()
            loss_sum += loss_total.item()
            # loss_s1 += si_loss1.item()
            # loss_s2 += si_loss2.item()

            # 输出损失信息
            # log('Training: epoch: %d, step:%d, ls_reg: %.2f, total_loss: %.2f' \
            #     % (e, step, regression_loss.data.cpu().numpy(), loss_total))

        if validation_dataloader is not None:
            model.eval()
            for step, (features, labels, IDs) in enumerate(validation_dataloader):
                val_steps = val_steps + 1

                features = features.to(device)
                labels = labels.to(device)

                # raw_bcg = features[:, 0:-dim_PI - dim_FF]
                # covar = features[:, -dim_PI - dim_FF:]
                x = features
                last_BP = features[:, -2:]  # DBP, SBP

                regression_out, _, recon_bcg= model(x)

                regression_loss = loss_regression(regression_out, labels)
                reconstruction_loss = loss_reconstruction(recon_bcg.squeeze(1), x[:, 0:1250])


                if params['is_similarity_loss']:
                    pass # loss_val = regression_loss.item() + si_loss1.item() + si_loss2.item()
                else:
                    loss_val = regression_loss.item() + reconstruction_loss.item()
                loss_val_sum += loss_val

                # log('Validation: epoch: %d, step:%d, ls_reg: %.2f, total_loss: %.2f' \
                #     % (e, step, regression_loss.data.cpu().numpy(), loss_val))

            val_loss = loss_val_sum / val_steps
        else:
            val_loss = loss_sum / train_steps

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log('saving model')
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), model_path)

        log('BPEstimator: [%d/%d] loss: %.4f, val_loss: %.4f'
            % (e + 1, epoch, loss_sum / train_steps, val_loss))
        log('Cunrrent ephoch run time: %.3f s' % (time.time() - st))

        torch.cuda.empty_cache()

    return best_model