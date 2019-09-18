import random
import glob
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
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

def make_snippet(audiomix, audiovocal, pos, window_sizes):
    """Extract a snippet from audio channels, adding padding if required.

    Preconditions:
      - input_size >= output_size
      - input_size - output_size is divisible by 2
    Performs padding if needed:
      - if output_end > n_samples
      - if input_start < 0
      - if input_end > n_samples
    """
    n_samples = audiomix.size()[0]
    output_end = pos+window_sizes[1]
    overreach = int((window_sizes[0] - window_sizes[1]) / 2)
    input_start = pos-overreach
    input_end = output_end+overreach
    
    if input_start>=0 and input_end<=n_samples and output_end<=n_samples:
        return audiomix[input_start:input_end, :], audiovocal[pos:output_end, :]

    input_prepad = 0-input_start if input_start < 0 else 0
    input_postpad = input_end-n_samples if input_end>n_samples else 0
    output_postpad = output_end-n_samples if output_end>n_samples else 0

    paddedmix = F.pad(input=audiomix[input_start+input_prepad:input_end-input_postpad, :],
                      pad=(0, 0, input_prepad, input_postpad),
                      mode='constant', value=0)
    paddedvocal = F.pad(input=audiovocal[pos:output_end-output_postpad, :],
                        pad=(0, 0, 0, output_postpad),
                        mode='constant', value=0)

    return paddedmix, paddedvocal

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
        assert window_sizes[0] >= window_sizes[1]

    def __iter__(self):
        for audio_mix, audio_vocal in self.audio_iter:
            for _ in range(0, self.num_snippets):
                pos = random.randint(0, audio_mix.size()[0] - self.window_sizes[1])
                snippet = make_snippet(audio_mix, audio_vocal, pos, self.window_sizes)
                yield snippet

