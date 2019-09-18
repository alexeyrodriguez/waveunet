


This is a pytorch version of Wave-U-Net.

More to follow.

=== Data Preparation

Download the mudb18 dataset

Install the musdb18 package. The package will install an utility `stem2wav` to extract wav files
from the different channels of a `.stem` file.

The .stem files should have their channels extracted to individual .wav files:

>  for stemfile in musdb18/train/*; do ~/.local/bin/stem2wav "$stemfile" musdb18_wave; done

In order to run this utility you might need to install ffmpeg and library sndfile.

=== Testing

> python -m pytest
