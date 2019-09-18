import pytest
from datasets.audio import make_snippet
import torch

@pytest.fixture
def mix():
    return torch.randn(2, 1024)

@pytest.fixture
def vocal():
    return torch.randn(2, 1024)

class TestMakeSnippet:
    def test_same_window_size(self, mix, vocal):
        pos = 16
        snipmix, snipvocal = make_snippet(mix, vocal, pos, (8, 8))
        assert snipmix.equal(mix[:, pos:pos+8])
        assert snipvocal.equal(vocal[:, pos:pos+8])

    def test_larger_input_size(self, mix, vocal):
        pos = 16
        snipmix, snipvocal = make_snippet(mix, vocal, pos, (12, 8))
        assert snipmix.equal(mix[:, pos-2:pos+10])
        assert snipvocal.equal(vocal[:, pos:pos+8])

    def test_larger_input_left_padding(self, mix, vocal):
        pos = 0
        snipmix, snipvocal = make_snippet(mix, vocal, pos, (12, 8))
        padding = torch.zeros(2, 2)
        wanted_mix = torch.cat([padding, mix[:, pos:pos+10]], 1)
        assert snipmix.equal(wanted_mix)
        assert snipvocal.equal(vocal[:, pos:pos+8])

    def test_larger_input_right_padding(self, mix, vocal):
        pos = 1024-8
        snipmix, snipvocal = make_snippet(mix, vocal, pos, (12, 8))
        padding = torch.zeros(2, 2)
        wanted_mix = torch.cat([mix[:, pos-2:pos+8], padding], 1)
        assert snipmix.equal(wanted_mix)
        assert snipvocal.equal(vocal[:, pos:pos+8])

    def test_larger_input_right_padding2(self, mix, vocal):
        pos = 1024-6
        snipmix, snipvocal = make_snippet(mix, vocal, pos, (12, 8))
        padding_mix = torch.zeros(2, 4)
        wanted_mix = torch.cat([mix[:, pos-2:pos+6], padding_mix], 1)
        padding_vocal = torch.zeros(2, 2)
        wanted_vocal = torch.cat([vocal[:, pos:pos+6], padding_vocal], 1)
        assert snipmix.equal(wanted_mix)
        assert snipvocal.equal(wanted_vocal)


