import torch
import torch.nn as nn
import torch.nn.functional as F

class ResampleBlock(nn.Module):
    '''
    A Resample block groups a pair of downsampling and upsampling blocks together.
    The pair is for corresponding down and up blocks, the down block sends a tensor
    to the block below but also to its correspding up block.

    Down blocks do not correspond to those of Stoller et al.
    A down block starts at the decimation/downsampling operation and includes the subsequent convolution.
    This choice allows both the up and down blocks to share the same number of up and down channels
    which makes the interface of the ResampleBlock simpler.

    A ResampleBlock can also have the decimation/upsampling operations disabled, in this way
    we can use it to implement the two top level convolutions (also the output_channels in this case can be
    different if desired).

    Args:
        input_channels (int): the number of channels of the down block input.
        down_channels (int): the number of channels outgoing from the down block,
            which is also the number of channels going into the up block.
        down_kernel_size (int): filter size of down block convolution
        up_kernel_size (int): filter size of up block convolution
        inner_module (callable): module that consumes down block output and produces the input of the up block
        resample (boolean): whether to include the decimation and upsampling operations, default True
        output_channels (boolean): the number of channels outgoing from the up block, default None (same as input_channels)
    '''
    def __init__(self, input_channels, down_channels, down_kernel_size, up_kernel_size, inner_module,
            resample=True, output_channels=None, output_activation=F.leaky_relu):
        super(ResampleBlock, self).__init__()
        self.resample = resample
        output_channels = input_channels if not output_channels else output_channels
        self.output_activation = output_activation
        self.down_conv = nn.Conv1d(input_channels, down_channels, down_kernel_size)
        self.up_conv = nn.Conv1d(down_channels+input_channels, output_channels, up_kernel_size)
        self.inner_module = inner_module

    def forward(self, x):
        '''
        Applies a resampling transformation on x.
        Convolution operations usually require padding to transform borders,
        this module does not perform padding and instead produces an output smaller
        than the input.
        '''
        x, saved = self._down(x)
        x = self.inner_module(x)
        return self._up(x, saved)

    def _down(self, x):
        decimated = x[:, :, ::2] if self.resample else x
        return F.leaky_relu(self.down_conv(decimated)), x

    def _up(self, x, saved):
        upsampled = F.interpolate(x, x.size()[2]*2-1, mode='linear') if self.resample else x
        enriched = torch.cat([centered_crop(saved, upsampled), upsampled], 1)
        return self.output_activation(self.up_conv(enriched))

def centered_crop(t, target_shape):
    s, target_s = t.size()[2], target_shape.size()[2]
    d = s - target_s
    if d==0:
        return t
    return t[:,:,d//2:-d+d//2]
