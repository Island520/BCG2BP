import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_heads, n_layers, dropout):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.input_layer = nn.Linear(n_features, 512)  # map from input features to transformer size
        self.causal_conv1d = nn.Conv1d(512, 512, kernel_size=3, padding=2)
        self.pos_encoder = PositionalEncoding(512, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, n_heads, dropout=dropout), n_layers)
        self.output_layer = nn.Linear(512, n_features)  # map from transformer size back to input features
        self.fc = nn.Linear(377 * 7, 256)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.input_layer(src)
        src = self.causal_conv1d(src.permute(1, 2, 0)).permute(2, 0, 1)  # apply causal convolution

        # Update the attention mask
        device = src.device
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.output_layer(output)

        # Reshape output to [batch_size, 377*7]
        output = output.permute(1, 0, 2).reshape(output.size(1), -1)

        # Apply fully connected layer
        # output = self.fc(output)

        return output

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
    def __init__(self, output_transformer, output_SEM, hidden_dim,
                 output_dim):
        super(CombinedModel, self).__init__()
        self.dim_BCG = 200
        self.dim_PI = 4
        self.dim_FF = 44
        self.transformer_model = TransformerModel(n_features=1, n_heads=8, n_layers=2, dropout=0.1)
        self.static_enrichment_module = StaticEnrichmentModule(input_dim=48, hidden_dim=48)
        self.fc1 = nn.Linear(output_transformer + output_SEM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, BCG, static_covariates):
        BCG = BCG.permute(2, 0, 1)
        transformer_output = self.transformer_model(BCG)
        static_enrichment_output = self.static_enrichment_module(static_covariates)

        # Adjust the output shape
        # transformer_output = transformer_output.view(transformer_output.size(0), -1)
        static_enrichment_output = static_enrichment_output.view(static_enrichment_output.size(0), -1)

        combined = torch.cat((transformer_output, static_enrichment_output), dim=-1)
        hidden = F.relu(self.fc1(combined))
        output = self.fc2(hidden)
        return output


# BCG = torch.randn(32, 1, 375)
# covar = torch.randn(32, 1, 48)
# BCG = BCG.permute(2, 0, 1) # We reshape to [features=375, batch_size=32, channels=7]
# model = TransformerModel(n_features=1, n_heads=8, n_layers=2, dropout=0.1)
# output = model(BCG)



# model2 = StaticEnrichmentModule(input_dim=48, hidden_dim=128,output_dim=128)
# output2 = model2(covar)
# print(output2.shape)  # Should be [32, 2]

# covar = torch.randn(32,1,48)
# model2 = StaticEnrichmentModule(input_dim=48, hidden_dim=48)
# output2 = model2(covar)

# BPEstimator = CombinedModel(output_transformer=377, output_SEM=48, hidden_dim=48, output_dim=2)
# output = BPEstimator(BCG, covar)
# print(output.shape)  # Should be [32, 2]
