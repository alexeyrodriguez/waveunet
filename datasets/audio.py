import os
import random
import glob
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
import torchaudio
from torchaudio.compliance.kaldi import resample_waveform

def musdb18_basenames(path):
    """Get the base names of wav-separated musdb18 files."""
    return [fname[:-6] for fname in glob.glob(path + '/*_0.wav')]

def musdb18_audio(basenames, sampling_rate=None):
    """Load mix and vocal channels from wav-separated musdb18 songs."""
    for name in basenames:
        mix_name = name + '_0.wav'
        vocal_name = name + '_4.wav'
        audio_mix, msr = torchaudio.load(mix_name)
        audio_vocal, vsr = torchaudio.load(vocal_name)
        if sampling_rate:
            audio_mix = resample_waveform(audio_mix, msr, sampling_rate)
            audio_vocal = resample_waveform(audio_vocal, vsr, sampling_rate)
        else:
            assert msr==vsr       
        yield audio_mix, audio_vocal

def musdb18_transform(sampling_rate, window_sizes, transform_callable, target_path, fnames):
    audio_iter = musdb18_audio(fnames, sampling_rate)
    transform = lambda t: transform_callable(t.unsqueeze(0)).squeeze()
    transformed_audio_iter = AudioSnippetsDataset.audio_transform(audio_iter, window_sizes, transform)
    for fname, transformed_audio in zip(fnames, transformed_audio_iter):
        target_name = '{}/{}_generated.wav'.format(target_path, os.path.basename(fname))
        torchaudio.save(target_name, transformed_audio, sampling_rate)

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
    n_samples = audiomix.size()[1]
    output_end = pos+window_sizes[1]
    overreach = int((window_sizes[0] - window_sizes[1]) / 2)
    input_start = pos-overreach
    input_end = output_end+overreach
    
    if input_start>=0 and input_end<=n_samples and output_end<=n_samples:
        return audiomix[:, input_start:input_end], audiovocal[:, pos:output_end]

    input_prepad = 0-input_start if input_start < 0 else 0
    input_postpad = input_end-n_samples if input_end>n_samples else 0
    output_postpad = output_end-n_samples if output_end>n_samples else 0

    paddedmix = F.pad(input=audiomix[:, input_start+input_prepad:input_end-input_postpad],
                      pad=(input_prepad, input_postpad),
                      mode='constant', value=0)
    paddedvocal = F.pad(input=audiovocal[:, pos:output_end-output_postpad],
                        pad=(0, output_postpad),
                        mode='constant', value=0)

    return paddedmix, paddedvocal

class AudioSnippetsDataset(IterableDataset):
    """A Dataset of audio file snippets .
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
        ordered (boolean): Whether to extract snippets in order rather than randomly,
            in this case the entire audio file is sampled.
    """

    def __init__(self, audio_iter, window_sizes, num_snippets, ordered=False):
        self.audio_iter = audio_iter
        self.window_sizes = window_sizes
        self.num_snippets = num_snippets
        self.ordered = ordered
        assert window_sizes[0] >= window_sizes[1]
        if num_snippets is not -1 and ordered:
            raise ValueError('When ordered is True, num_snippets should be set to -1.')

    def __iter__(self):
        for audio_mix, audio_vocal in self.audio_iter:
            assert audio_mix.size()[1] >= self.window_sizes[1]
            for pos in self._position_iterator(audio_mix.size()[1]):
                yield make_snippet(audio_mix, audio_vocal, pos, self.window_sizes)

    @classmethod
    def audio_transform(cls, audio_iter, window_sizes, transform_callable):
        """Transform audio iterators by applying a callable to ordered mix snippets.
           This is truly an ugly function.
        """
        for audio_mix, audio_vocal in audio_iter:
            num_audio_samples = audio_mix.size()[1]
            snippet_iterator = cls([(audio_mix, audio_vocal)], window_sizes, -1, True)
            transformed_snippets = (transform_callable(snippet_mix) for snippet_mix, _ in snippet_iterator)
            padded_output = torch.cat(list(transformed_snippets), 1)
            yield padded_output[:, 0:num_audio_samples]

    def _position_iterator(self, num_audio_samples):
        if not self.ordered:
            for _ in range(0, self.num_snippets):
                yield random.randint(0, num_audio_samples - self.window_sizes[1])
        else:
            for pos in range(0, num_audio_samples, self.window_sizes[1]):
                yield pos


