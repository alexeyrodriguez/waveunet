import torch
import torch.nn as nn
import torch.nn.functional as F

class ResampleBlock(nn.Module):
    def __init__(self, input_channels, down_channels, down_kernel_size, up_kernel_size, inner_module, resample=True, output_channels=None):
        super(ResampleBlock, self).__init__()
        self.resample = resample
        output_channels = input_channels if not output_channels else output_channels
        self.down_conv = nn.Conv1d(input_channels, down_channels, down_kernel_size)
        self.up_conv = nn.Conv1d(down_channels+input_channels, output_channels, up_kernel_size)
        self.inner_module = inner_module

    def forward(self, x):
        x, saved = self._down(x)
        x = self.inner_module(x)
        return self._up(x, saved)

    def _down(self, x):
        decimated = x[:, :, ::2] if self.resample else x
        return self.down_conv(decimated), x

    def _up(self, x, saved):
        upsampled = F.interpolate(x, x.size()[2]*2-1, mode='linear') if self.resample else x
        enriched = torch.cat([centered_crop(saved, upsampled), upsampled], 1)
        return self.up_conv(enriched)

def centered_crop(t, target_shape):
    s, target_s = t.size()[2], target_shape.size()[2]
    d = s - target_s
    if d==0:
        return t
    return t[:,:,d//2:-d+d//2]
