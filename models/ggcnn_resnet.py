import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin,cin,7,1, padding=3), nn.ReLU())

    def forward(self, x):
        y = self.conv(x) + x
        return y


class ResNet(nn.Module):
    def __init__(self, input_channels=1):
        super(ResNet, self).__init__()
        self.conv_in = nn.Sequential(nn.BatchNorm2d(input_channels), nn.Conv2d(input_channels,64,7,1, padding=3),
                                        ResBlock(64),
                                        ResBlock(64),
                                        ResBlock(64),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(64), nn.Conv2d(64,128,3,1, padding=1),
                                        ResBlock(128),
                                        ResBlock(128),
                                        ResBlock(128),
                                        nn.MaxPool2d(2,2))

        self.conv_o  = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                     ResBlock(128),
                                     ResBlock(128),
                                     ResBlock(128),
                                     nn.BatchNorm2d(128), nn.Conv2d(128,64,3,1, padding=1),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     ResBlock(64),
                                     ResBlock(64),
                                     ResBlock(64),
                                     nn.BatchNorm2d(64), )

        self.pos_output = nn.Conv2d(64,1,7,1, padding=3)
        self.cos_output = nn.Conv2d(64,1,7,1, padding=3)
        self.sin_output = nn.Conv2d(64,1,7,1, padding=3)
        self.width_output = nn.Conv2d(64,1,7,1, padding=3)

    def forward(self, x):
        z = self.conv_in(x)
        y = self.conv_o (z)

        pos_output = torch.sigmoid(self.pos_output(y))
        cos_output = self.cos_output(y)
        sin_output = self.sin_output(y)
        width_output = self.width_output(y)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
