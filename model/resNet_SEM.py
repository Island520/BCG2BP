import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resNet(nn.Module):
    def __init__(self, num_classes=128):
        super(resNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入通道为1
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 3)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride=1):
        layers = []
        in_planes = 64
        for _ in range(num_blocks):
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class StaticEnrichmentModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StaticEnrichmentModule, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, a):
        # a = a.view(a.size(0), -1)  # Flatten the input
        hid1 = F.elu(self.W1(a))
        hid2 = self.W2(hid1)

        out_GLU = torch.sigmoid(self.W3(hid2)) * self.W4(hid2)

        return self.LayerNorm(a + out_GLU)


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        self.dim_BCG = 200
        self.dim_PI = 4
        self.dim_FF = 44

        self.resNet = resNet()
        self.static_enrichment_module = StaticEnrichmentModule(input_dim=48, hidden_dim=48)
        self.regressor = nn.Sequential(
            nn.Linear(176, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, 2)
        )

    def forward(self, BCG, static_covariates):
        BCG = BCG.reshape(BCG.shape[0], 1, BCG.shape[1])
        resNet_output = self.resNet(BCG)
        static_enrichment_output = self.static_enrichment_module(static_covariates)

        # Adjust the output shape
        # transformer_output = transformer_output.view(transformer_output.size(0), -1)
        static_enrichment_output = static_enrichment_output.view(static_enrichment_output.size(0), -1)

        combined = torch.cat((resNet_output, static_enrichment_output), dim=-1)
        output = self.regressor(combined)

        return output
