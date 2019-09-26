from . import ResampleBlock
import torch.nn as nn

class WaveUNet(nn.Module):
    '''
    Creates a WaveUNet for source separation as described by Stoller et al.

    Args:
        input_channels (int): number of channels in input
        output_channels (int): number of channels in output
        down_kernel_size (int): kernel size used in down convolutions
        up_kernel_size (int): kernel size used in up convolutions
        depth (int): number of pairs of down and up blocks
        num_filters (int): number of additional convolution channels used at each deeper level
    '''
    def __init__(self, input_channels, output_channels, down_kernel_size, up_kernel_size, depth, num_filters):
        super(WaveUNet, self).__init__()

        # Create Resample blocks in a bottom to top direction
        block_stack = lambda x: x
        for i in range(depth):
            up_channels = (depth - i) * num_filters
            down_channels = (depth - i + 1) * num_filters
            block_stack = ResampleBlock(up_channels, down_channels, down_kernel_size, up_kernel_size, block_stack)

        self.top_block = ResampleBlock(input_channels, num_filters, down_kernel_size, 1, block_stack, resample=False, output_channels=output_channels)

    def forward(self, x):
        '''
        Applies a WaveUNet transformation to input tensor.
        Convolutions require context due to not performing padding when convolving borders (i.e. borders are not convolved :) ).
        Therefore the input is usually larger than the output, the difference depends on filter sizes and depth,
        see WaveUNetSizeCalculator to calculate the exact sizes.
        '''
        return self.top_block(x)
