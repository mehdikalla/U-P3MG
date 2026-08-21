import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, channels_in, channels_out, batch_norm, dropout: float = 0.0):
        super(BasicConv, self).__init__()
        conv = nn.Conv1d(channels_in, channels_out,
                          kernel_size=3, stride=1, padding=1, bias=True)
        # Initialisation Kaiming (He), approximee via 'leaky_relu' (pas de mode dedie a PReLU).
        nn.init.kaiming_normal_(conv.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(conv.bias)

        basic_conv = [conv, nn.PReLU()]
        if batch_norm:
            basic_conv.append(nn.BatchNorm1d(channels_out))
        if dropout > 0.0:
            basic_conv.append(nn.Dropout(dropout))

        self.body = nn.Sequential(*basic_conv)

    def forward(self, x):
        return self.body(x)


class ResUNetConv(nn.Module):
    def __init__(self, num_convs, channels, batch_norm, dropout: float = 0.0):
        super(ResUNetConv, self).__init__()
        unet_conv = []
        for _ in range(num_convs):
            conv = nn.Conv1d(channels, channels,
                             kernel_size=3, stride=1, padding=1, bias=True)
            nn.init.kaiming_normal_(conv.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(conv.bias)
            unet_conv.append(conv)
            unet_conv.append(nn.PReLU())
            if batch_norm:
                unet_conv.append(nn.BatchNorm1d(channels))
            if dropout > 0.0:
                unet_conv.append(nn.Dropout(dropout))

        self.body = nn.Sequential(*unet_conv)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class ResUNetBlock(nn.Module):
    """
    Bloc encodeur du ResU-Net.

    Applique une convolution de projection de canaux suivie d'un bloc
    residuel, puis un sous-echantillonnage par max-pooling. La sortie
    pre-pooling est egalement renvoyee afin de servir de connexion de saut
    (skip connection) pour le decodeur symetrique.
    """

    def __init__(self, channels_in, channels_out, num_convs, batch_norm, dropout: float = 0.0):
        super(ResUNetBlock, self).__init__()
        self.conv1 = BasicConv(channels_in, channels_out, batch_norm, dropout)
        self.resunet_conv = ResUNetConv(num_convs, channels_out, batch_norm, dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.resunet_conv(self.conv1(x))
        pooled = self.pool(skip)
        return pooled, skip


class ResUNetUpBlock(nn.Module):
    """
    Bloc decodeur du ResU-Net.

    Sur-echantillonne l'entree par interpolation lineaire, la concatene a la
    connexion de saut correspondante issue de l'encodeur, puis applique une
    convolution de projection de canaux suivie d'un bloc residuel.
    """

    def __init__(self, channels_in, channels_skip, channels_out, num_convs, batch_norm, dropout: float = 0.0):
        super(ResUNetUpBlock, self).__init__()
        self.conv1 = BasicConv(channels_in + channels_skip, channels_out, batch_norm, dropout)
        self.resunet_conv = ResUNetConv(num_convs, channels_out, batch_norm, dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = tc.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.resunet_conv(x)
        return x

