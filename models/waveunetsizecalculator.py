import numpy as np
from numpy.linalg import multi_dot, inv

class WaveUNetSizeCalculator:
    '''
    Calculates input and output size based on a requested output size.
    The calculated input size is larger than the output to include the context needed
    to convolve the borders.
    Plus the upsampling assumes that its inputs must be odd, so that constraint is also
    enforced.
    '''
    def __init__(self, downblock_kernel_size, upblock_kernel_size, depth):
        self.downblock_kernel_size = downblock_kernel_size
        self.upblock_kernel_size = upblock_kernel_size
        self.depth = depth
 
    def calculate_dimensions(self, requested_output):
        downsample = [self._down_conv()] + [self._down()] * self.depth
        upsample = [self._up()] * self.depth + [np.eye(2)]

        # Input size needed before up blocks to produce requested output
        # Truncate in order to satisfy oddness constraint in upsampling
        input_pre_upblocks = _multi_dot(upsample).dot(np.array([requested_output, 1]))
        input_pre_upblocks = np.floor(input_pre_upblocks)

        output = inv(_multi_dot(upsample)).dot(input_pre_upblocks.T)
        input = _multi_dot(downsample).dot(input_pre_upblocks.T)

        _sanity_check(downsample + upsample, input, output)

        return (int(input[0]), int(output[0]))

    # Equations between input and output sizes are expressed as linear transformations
    # The linear transform receives a vector [output, 1] and produces [input, 1]
    # To be used with care as arithmetic should be carried in the integer domain
  
    def _down(self):
        return self._down_conv().dot(self._down_decimation())

    def _up(self):
        return self._up_upsample().dot(self._up_conv())

    def _down_conv(self):
        'input = output + downblock_kernel_size - 1'
        return np.array([[1, self.downblock_kernel_size-1], [0, 1]])

    def _down_decimation(self):
        'input = output * 2'
        return np.array([[2, 0], [0, 1]])

    def _up_upsample(self):
        '''
        input = output / 2 + 0.5, equivalent to
        input * 2 + 1 = output, on integer domain constrains output to be odd
        '''
        return np.array([[0.5, 0.5], [0, 1]])

    def _up_conv(self):
        'input = output + upblock_kernel_size - 1'
        return np.array([[1, self.upblock_kernel_size-1], [0, 1]])

def _sanity_check(ms, input, output):
    '''Check that equations hold in integer domain.'''
    t = output
    for m in reversed(ms):
        t = m.dot(t)
        assert np.array_equal(t, np.floor(t))
    assert np.array_equal(t, input)

def _multi_dot(ms):
    return ms[0] if len(ms)==1 else multi_dot(ms)
