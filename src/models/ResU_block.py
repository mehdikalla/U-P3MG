import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, channels_in, channels_out, batch_norm):
        super(BasicConv, self).__init__()
        basic_conv = [nn.Conv1d(channels_in, channels_out,
                                kernel_size=3, stride=1, padding=1, bias=True)]
        basic_conv.append(nn.PReLU())
        if batch_norm:
            basic_conv.append(nn.BatchNorm1d(channels_out))

        self.body = nn.Sequential(*basic_conv)

    def forward(self, x):
        return self.body(x)


class ResUNetConv(nn.Module):
    def __init__(self, num_convs, channels, batch_norm):
        super(ResUNetConv, self).__init__()
        unet_conv = []
        for _ in range(num_convs):
            unet_conv.append(nn.Conv1d(channels, channels,
                             kernel_size=3, stride=1, padding=1, bias=True))
            unet_conv.append(nn.PReLU())
            if batch_norm:
                unet_conv.append(nn.BatchNorm1d(channels))

        self.body = nn.Sequential(*unet_conv)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class ResUNetBlock(nn.Module):
    def __init__(self, channels_in, channels_out, num_convs, batch_norm):
        super(ResUNetBlock, self).__init__()
        self.conv1 = BasicConv(channels_in, channels_out, batch_norm)
        self.resunet_conv = ResUNetConv(num_convs, channels_out, batch_norm)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resunet_conv(x)
        x = self.pool(x)
        return x