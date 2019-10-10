
# Wave-U-Net PyTorch reimplementation

This is a PyTorch reimplementation of Wave-U-Net as described by
[Stoller at al. / Wave-U-Net: A Multi-Scale Neural Network for End-To-End Audio Source Separations](https://arxiv.org/abs/1806.03185).
The paper authors implementation can be found [here](https://github.com/f90/Wave-U-Net).

This repository does not implement all the variants. The selected variant includes additional input context (no zero padding),
upsampling through linear interpolation (no learned upsampling) and supports stereo channels and multiple source extraction.

## Data Preparation

Download the [musdb18 dataset](https://sigsep.github.io/datasets/musdb.html)

Install the musdb18 package. The package will install an utility `stem2wav` to extract wav files
from the different channels of a `.stem` file.

The `.stem` files should have their channels extracted to individual `.wav` files:

```
for stemfile in musdb18/train/*; do ~/.local/bin/stem2wav "$stemfile" musdb18_wave; done
```

In order to run `stem2wav` you might need to install ffmpeg and library sndfile.

# Training

You will need to update `config.yml` to match your path setup.

# TODO

+ Implement multiple source extraction rather than just vocal extraction
+ Preprocessing and shuffling input data for better training performance and runtime?

# Unit testing

```
python -m pytest
```
