import torch
from torch import nn


class VideoClassifier(nn.Module):
    """Video Classifier Based on Transformer"""

    def __init__(
        self,
        input_dim,
        output_dim,
        in_channels=1,
        out_channels=10,
        kernel_size=10,
        stride=10,
        num_layers=6,
        nhead=2,
    ) -> None:
        super(VideoClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=input_dim)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pool = nn.AvgPool1d(kernel_size=out_channels)
        self.lin = nn.Linear(in_features=input_dim // stride, out_features=output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        bn_X = self.bn1(X)
        bn_X = bn_X.permute(0, 2, 1)
        conv_X = self.conv1d(bn_X)
        conv_X = conv_X.permute(0, 2, 1)
        trans_X = self.encoder(conv_X)
        pool_X = self.pool(trans_X).squeeze(dim=-1)
        lin_X = self.lin(pool_X)
        y = self.softmax(lin_X)
        return y
