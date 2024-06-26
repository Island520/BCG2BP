import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from model.U2net import *

# ! V0 -> V0a
# * adjusted kernelsize and padding in TemporalResBlock
class TemporalResBlock(nn.Module):
    def __init__(self, inCh, outCh):
        super(TemporalResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=inCh, out_channels=inCh, kernel_size=7, padding=3, groups=inCh)
        self.bn1 = nn.BatchNorm1d(num_features=inCh)
        self.relu1 = nn.ReLU()

        self.l1 = nn.Linear(inCh, outCh)
        self.relu2 = nn.ReLU()

        self.l2 = nn.Linear(outCh, outCh)
        self.bn3 = nn.BatchNorm1d(num_features=outCh)

        self.convres = nn.Conv1d(in_channels=inCh, out_channels=outCh, kernel_size=1)
        self.bnres = nn.BatchNorm1d(num_features=outCh)

        self.reluout = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = y.permute(0, 2, 1)
        y = self.l1(y)
        y = self.relu2(y)
        y = self.l2(y)
        y = y.permute(0, 2, 1)
        y = self.bn3(y)

        res = self.convres(x)
        res = self.bnres(res)

        out = self.reluout(y + res)
        out = self.pool(out)

        return out


class TemporalBlock(nn.Module):
    def __init__(self):
        super(TemporalBlock, self).__init__()

        self.tresblock0 = TemporalResBlock(inCh=1, outCh=16)
        self.tresblock1 = TemporalResBlock(inCh=16, outCh=32)
        self.tresblock2 = TemporalResBlock(inCh=32, outCh=64)
        self.tresblock3 = TemporalResBlock(inCh=64, outCh=128)
        self.tresblock4 = TemporalResBlock(inCh=128, outCh=128)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1).permute(0, 2, 1)
        y = self.tresblock0(x)
        y = self.tresblock1(y)
        y = self.tresblock2(y)
        y = self.tresblock3(y)
        y = self.tresblock4(y)
        y = y.permute(0, 2, 1)

        return y


class GenSignalFeatures(nn.Module):
    def __init__(self):
        super(GenSignalFeatures, self).__init__()

        self.temporalblock = TemporalBlock()

        self.t_gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.t_bn1 = nn.BatchNorm1d(num_features=39)

    def forward(self, signal):
        feature_t = self.temporalblock(signal)

        (o0, h0) = self.t_gru(feature_t)
        y = torch.squeeze(o0)
        y = self.t_bn1(y)

        return y


class BCGNet(nn.Module):

    def __init__(self):
        super(BCGNet, self).__init__()

        self.dim_BCG = 1250
        self.dim_PI = 4
        self.dim_FF = 44
        self.dim_LAST_BP = 0

        self.bcg_feature_gen = U2Net(1,1)
        # self.ecg_feature_gen = GenSignalFeatures()

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(79, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        raw_bcg = x[:, 0:1250]
        covar = x[:, -self.dim_PI - self.dim_FF:]
        last_BP = x [:, -2:]

        recon_bcg, code = self.bcg_feature_gen(raw_bcg)
        # ecg_t = self.ecg_feature_gen(x[:, :, 1])
        mid_feature = torch.mean(code, 1)
        # feature = torch.cat((
        #     torch.mean(ppg_t, 1),
        #     torch.mean(ecg_t, 1)
        # ),
        #     dim=-1)
        if self.dim_LAST_BP:
            out = self.dense(torch.cat((mid_feature,last_BP), dim=-1))
        else:
            out = self.dense(mid_feature)

        return out, mid_feature, recon_bcg

    def requested_input_columns(self) -> list:
        return ['sig']

    def set_tunable_layers(self, tune):
        for p in self.parameters():
            p.requires_grad = True
        if tune == "all":
            return
        elif tune=='default':
            for p in self.bcg_feature_gen.parameters():
                p.requires_grad = False
        else:
            raise Exception("undefined tune")


hooked = dict()


def fhook(self, layer_input, layer_output):
    hooked[self] = layer_output


if __name__ == "__main__":
    input = torch.rand((128, 1250))
    net = BCGNet()
    num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(net)
    print("has {:10d} trainable params".format(num_trainable_params))

    net.dense[0].register_forward_hook(fhook)
    output = net(input)
    print(output.shape)

    print(net.dense)

    for key, val in hooked.items():
        print(key)
        print(val.shape)












