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
        if sampling_rate and msr!=sampling_rate:
            audio_mix = resample_waveform(audio_mix, msr, sampling_rate)
            audio_vocal = resample_waveform(audio_vocal, vsr, sampling_rate)
        else:
            assert msr==vsr       
        yield audio_mix, audio_vocal

def musdb18_transform(sampling_rate, window_sizes, device, transform_callable, target_path, fnames):
    audio_iter = (audio for audio, _ in musdb18_audio(fnames, sampling_rate))
    transform = lambda t: transform_callable(t.unsqueeze(0)).squeeze()
    transformed_audio_iter = audio_transform(audio_iter, window_sizes, device, transform)
    for fname, transformed_audio in zip(fnames, transformed_audio_iter):
        target_name = '{}/{}_vocals.wav'.format(target_path, os.path.basename(fname))
        torchaudio.save(target_name, transformed_audio, sampling_rate)

def make_audio_snippet(audio, start, end):
    """
    Extract a snippet from audio, padded with zeros if `start` or `end` go beyond the audio sample
    """
    n_samples = audio.size()[1]
    prepad, postpad = max(0-start, 0), max(end-n_samples, 0)
    if prepad==0 and postpad==0:
        return audio[:, start:end]
    else:
        return F.pad(input=audio[:, start+prepad:end-postpad],
                     pad=(prepad, postpad),
                     mode='constant', value=0)

def make_snippet(audiomix, audiovocal, pos, window_sizes):
    """Extract a snippet from audio input channels, adding padding if required.

    Preconditions:
      - input_size >= output_size
      - input_size - output_size is divisible by 2
      - input_size >= output_size
    Performs padding if needed:
      - if input_start < 0
      - if input_end > n_samples
      - if output_end > n_samples
    """
    return (make_input_snippet(audiomix, pos, window_sizes),
            make_audio_snippet(audiovocal, pos, pos+window_sizes[1]))

def make_input_snippet(audiomix, pos, window_sizes):
    input_overreach = (window_sizes[0] - window_sizes[1]) // 2
    return make_audio_snippet(audiomix, pos-input_overreach, pos+window_sizes[1]+input_overreach)

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

    def _position_iterator(self, num_audio_samples):
        if not self.ordered:
            for _ in range(0, self.num_snippets):
                yield random.randint(0, num_audio_samples - self.window_sizes[1])
        else:
            for pos in range(0, num_audio_samples, self.window_sizes[1]):
                yield pos

def audio_transform(audio_iter, window_sizes, device, transform_callable):
    """
    Transform audio iterators by applying a callable to ordered mix snippets.
    """
    for audio_mix in audio_iter:
        snippets = []
        for pos in range(0, audio_mix.size()[1], window_sizes[1]):
            snippet = make_input_snippet(audio_mix, pos, window_sizes)
            snippet = transform_callable(snippet.to(device))
            snippets.append(snippet.cpu().detach())

        yield torch.cat(snippets, 1)[:, 0:audio_mix.size()[1]]
