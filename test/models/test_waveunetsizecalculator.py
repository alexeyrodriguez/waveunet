import pytest
from models import WaveUNetSizeCalculator
import numpy as np

k1 = 15 # Down conv kernel size
k2 = 5 # Up conv kernel size
i = 100 # Input size

@pytest.fixture
def wavecalc():
    return WaveUNetSizeCalculator(k1, k2, 3)

def check_eq(m, i, o):
    ti = np.array([i, 1])
    to = m.dot(np.array([o, 1])).T
    assert np.array_equal(ti, to), 'Transformed output {} does not match input {}'.format(to, ti)

class TestWaveUNetSizeCalculator:
    def test_down_conv(self, wavecalc):
        check_eq(wavecalc._down_conv(), i, i - k1 + 1)

    def test_down_decimation(self, wavecalc):
        check_eq(wavecalc._down_decimation(), i, i // 2)

    def test_up_upsample(self, wavecalc):
        check_eq(wavecalc._up_upsample(), i, i * 2 - 1)

    def test_up_conv(self, wavecalc):
        check_eq(wavecalc._up_conv(), i, i - k2 + 1)

    def test_calculation(self, wavecalc):
        assert wavecalc.calculate_dimensions(1024) == (1168, 1021)
