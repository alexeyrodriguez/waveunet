import random
import glob
from torch.utils.data import Dataset, IterableDataset
import torchaudio

def musdb18_basenames(path):
    """Get the base names of wav-separated musdb18 files."""
    return [fname[:-6] for fname in glob.glob(path + '/*_0.wav')]

def musdb18_audio(basenames):
    """Load mix and vocal channels from wav-separated musdb18 songs."""
    for name in basenames:
        mix_name = name + '_0.wav'
        vocal_name = name + '_1.wav'
        audio_mix, msr = torchaudio.load(mix_name, channels_first=False)
        audio_vocal, vsr = torchaudio.load(vocal_name, channels_first=False)
        assert msr==vsr       
        yield audio_mix, audio_vocal

# New snippets dataset
def make_snippet(audiomix, audiovocal, pos, window_sizes):
    """Extract a snippet from audio channels."""
    return audiomix[pos:pos+window_sizes[0], :], audiovocal[pos:pos+window_sizes[1], :]

class AudioSnippetsDataset(IterableDataset):
    """A Dataset of random snippets of audio files.
       The snippets are randomly sampled from audio files however, the file ordering
       is preserved and hence it might influence training negatively.
    
    Args:
        audio_iter (iterable of tensor pairs): The first component is the
            complete mix, the second component is the vocal channel.
            The dimensionality of each tensor is: #samples x #channels.
        window_sizes (pair of window sizes, respectively input and output):
            The snippets will have the corresponding sizes, zero padding will be added
            if necessary. The input size is required to be larger or equal to the
            output size. Input center is aligned to output center.
        num_snippets (integer): number of snippets per file
    """

    def __init__(self, audio_iter, window_sizes, num_snippets):
        self.audio_iter = audio_iter
        self.window_sizes = window_sizes
        self.num_snippets = num_snippets

    def __iter__(self):
        for audio_mix, audio_vocal in self.audio_iter:
            for _ in range(0, self.num_snippets):
                pos = random.randint(0, audio_mix.size()[0] - self.window_sizes[1])
                snippet = make_snippet(audio_mix, audio_vocal, pos, self.window_sizes)
                yield snippet

