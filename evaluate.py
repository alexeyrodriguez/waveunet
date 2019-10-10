import argparse

import torch

import config as cfg
import modelmgt
from datasets.audio import musdb18_transform, musdb18_basenames

def main():
    parser = argparse.ArgumentParser(description='Apply Source Separation')
    parser.add_argument('--model-path', help='Path to saved model checkpoint', required=True)
    parser.add_argument('--target-dir', help='Directory where to store extracted sources', required=True)
    parser.add_argument('audio_files', help='Input audio files to which to apply source separation', nargs='+')
    args = parser.parse_args()

    config = cfg.load(args.model_path + '/config.yml')
    window_sizes, model = modelmgt.create_waveunet_model(config)
    modelmgt.load_model(args.model_path, model)

    device = torch.device(config['device'])
    model.to(device)

    musdb18_transform(config['sampling_rate'], window_sizes, device, model, args.target_dir, musdb18_basenames(args.audio_files[0]))

if __name__ == '__main__':
    main()
