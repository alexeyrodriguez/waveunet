import argparse
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


from datasets.audio import make_snippet, AudioSnippetsDataset
from datasets.audio import AudioSnippetsDataset, musdb18_basenames, musdb18_audio, musdb18_transform
import models
import config as cfg

def training_fnames(config):
    train_names = musdb18_basenames(config['training_path'])
    sep_index = int(len(train_names) * config['validation_proportion'])
    random.shuffle(train_names)
    val_names, train_names = train_names[0:sep_index], train_names[sep_index:]
    return train_names,val_names

def train(optimizer, log_interval, epoch, device, model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Epoch: {:4d}\tBatch: {}\tTraining Loss: {}\t'.format(epoch, batch_idx, loss))

def evaluate(epoch, device, model, val_loader):
    model.eval()
    loss = 0.0
    batches = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.mse_loss(output, target)
            batches += 1
    loss = loss / batches
    print('Epoch: {:4d}\tValidation loss: {}'.format(epoch, loss))

def audio_snippets_loader(config, window_sizes, file_names):
    audio_snippets = AudioSnippetsDataset(musdb18_audio(file_names, config['sampling_rate']), window_sizes, config['snippets_per_audio_file'])
    return DataLoader(audio_snippets, config['batch_size'])

def main():
    parser = argparse.ArgumentParser(description='Source Separation Trainer')
    parser.add_argument('--config', help='Path to configuration file', required=True)
    args = parser.parse_args()

    config = cfg.load(args.config)
    window_sizes, model = models.waveunet(config['output_size'], 2, 2, config['down_kernel_size'], config['up_kernel_size'], config['depth'], config['num_filters'])
    print(window_sizes)

    device = torch.device(config['device'])
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    train_names, val_names = training_fnames(config)

    for epoch in range(config['training_epochs']):
        train(optimizer, config['batches_report'], epoch, device, model, audio_snippets_loader(config, window_sizes, train_names))

        if (epoch+1) % config['validation_epochs_frequency'] == 0:
            evaluate(epoch, device, model, audio_snippets_loader(config, window_sizes, val_names))
            print('Epoch: {:4d}\tApplying model to {} files.'.format(epoch, len(val_names)))
            musdb18_transform(config['sampling_rate'], window_sizes, device, model, config['generated_path'], val_names)

if __name__ == '__main__':
    main()
