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

def train(optimizer, log_interval, epoch, model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Epoch: {:4d}\tBatch: {}\tTraining Loss: {}\t'.format(epoch, batch_idx, loss))

def evaluate(epoch, model, val_loader):
    model.eval()
    loss = 0.0
    batches = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss += F.mse_loss(output, target)
            batches += 1
    loss = loss / batches
    print('Epoch: {:4d}\tValidation loss: {}'.format(epoch, loss))

def audio_snippets_loader(window_sizes, batch_size, file_names, sampling_rate):
    audio_snippets = AudioSnippetsDataset(musdb18_audio(file_names, sampling_rate), window_sizes, 2)
    return DataLoader(audio_snippets, batch_size)

def main():
    parser = argparse.ArgumentParser(description='Source Separation Trainer')
    parser.add_argument('--config', help='Path to configuration file', required=True)
    args = parser.parse_args()

    config = cfg.load(args.config)
    window_sizes, model = models.select('naive_regressor')
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_names, val_names = training_fnames(config)

    for epoch in range(2):
        train(optimizer, config['batches_report'], epoch, model, audio_snippets_loader(window_sizes, config['batch_size'], train_names, config['sampling_rate']))

        if epoch % 1 == 0 or True:
            evaluate(epoch, model, audio_snippets_loader(window_sizes, config['batch_size'], val_names, config['sampling_rate']))
            print('Epoch: {:4d}\tApplying model to {} files.'.format(epoch, len(val_names)))
            musdb18_transform(config['sampling_rate'], window_sizes, lambda x: x, config['generated_path'], val_names)

if __name__ == '__main__':
    main()
