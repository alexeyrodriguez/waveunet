from models import naiveregressor
from models.waveunetsizecalculator import WaveUNetSizeCalculator
from models.resampleblock import ResampleBlock
from models.waveunet import WaveUNet

def waveunet(requested_output, input_channels, output_channels, down_kernel_size, up_kernel_size, depth, num_filters):
    calc = WaveUNetSizeCalculator(down_kernel_size, up_kernel_size, depth)
    w = WaveUNet(input_channels, output_channels, down_kernel_size, up_kernel_size, depth, num_filters)
    return calc.calculate_dimensions(requested_output), w
