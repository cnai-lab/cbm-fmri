import torch.nn as nn
import torch
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder).__init__()
        self.conv = nn.Conv1d(in_channels=264, out_channels=40, kernel_size=3)
        self.deconv = nn.ConvTranspose1d(in_channels=40, out_channels=264, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


