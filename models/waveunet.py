from . import ResampleBlock
import torch.nn as nn

class WaveUNet(nn.Module):
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
        return self.top_block(x)
