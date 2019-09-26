import pytest
from models import ResampleBlock
import torch

i = 10
k_down = 5
k_up = 3

@pytest.fixture
def t1():
    return torch.arange(1, i*2*2+1, dtype=torch.float32).view(2, 2, i)

class TestResampleBlock:
    def test_down_conv(self, t1):
        r = ResampleBlock(2, 3, k_down, 1, lambda x: x, resample=False)
        assert r(t1).size()[2] == i - k_down + 1
        assert r(t1).size()[1] == 2

    def test_up_conv(self, t1):
        r = ResampleBlock(2, 3, 1, k_up, lambda x: x, resample=False)
        assert r(t1).size()[2] == i - k_up + 1
        assert r(t1).size()[1] == 2

    def test_resampling(self, t1):
        r = ResampleBlock(2, 3, 1, 1, lambda x: x, resample=True)
        assert r(t1).size()[2] == i - 1
        assert r(t1).size()[1] == 2

    def test_output_channels(self, t1):
        r = ResampleBlock(2, 3, 1, 1, lambda x: x, resample=True, output_channels=10)
        assert r(t1).size()[2] == i - 1
        assert r(t1).size()[1] == 10
