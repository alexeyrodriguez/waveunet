import pytest
from models import WaveUNet
from models import WaveUNetSizeCalculator
import torch

i = 40
k_down = 5
k_up = 3

@pytest.fixture
def t1():
    return torch.arange(1, i*2*2+1, dtype=torch.float32).view(2, 2, i)

class TestWaveUNet:
    def test_depth_0(self, t1):
        w = WaveUNet(2, 2, k_down, k_up, 0, 24)
        assert w(t1).size()[2] == i - k_down + 1

    def test_depth_1(self, t1):
        w = WaveUNet(2, 2, k_down, k_up, 1, 24)
        assert w(t1).size()[2] == 25

    def test_depth_5(self):
        c = WaveUNetSizeCalculator(k_down, k_up, 5)
        w = WaveUNet(2, 2, k_down, k_up, 5, 24)
        input_size, output_size = c.calculate_dimensions(1024)
        print((input_size, output_size))
        t = torch.randn((2, 2, input_size))
        assert w(t).size()[2] == output_size
