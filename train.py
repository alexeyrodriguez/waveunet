import argparse
import random
import datetime
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from datasets.audio import make_snippet, AudioSnippetsDataset
from datasets.audio import AudioSnippetsDataset, musdb18_basenames, musdb18_audio, musdb18_transform
import models
import config as cfg

def training_fnames(config):
    train_names = musdb18_basenames(config['training_path'])
    sep_index = int(len(train_names) * config['validation_proportion'])
    val_names, train_names = train_names[0:sep_index], train_names[sep_index:]
    return train_names,val_names

def train(report, optimizer, log_interval, epoch, device, model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        report.add_scalar('train_mse', loss)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Epoch: {:4d}\tBatch: {}\tTraining Loss: {}\t'.format(epoch, batch_idx, loss))

        report.advance_step()

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
    return loss

def audio_snippets_loader(round, config, window_sizes, file_names):
    audio_snippets = AudioSnippetsDataset(musdb18_audio(file_names, config['sampling_rate']), window_sizes, config['snippets_per_audio_file'])
    return DataLoader(audio_snippets, config['batch_size_{}'.format(round)])

def output_dirs(config):
    name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    base_dir = config['output_path']
    out_dir = base_dir + '/' + name
    pred_dir = out_dir + '/preds'
    log_dir = out_dir + '/log_dir'
    for dir in [base_dir, out_dir, pred_dir, log_dir]:
        os.makedirs(dir, exist_ok=True)
    return out_dir, pred_dir, log_dir

def save_checkpoint(fname, epoch, model, optimizer, eval_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eval_loss': eval_loss
        }, fname)

def main():
    parser = argparse.ArgumentParser(description='Source Separation Trainer')
    parser.add_argument('--config', help='Path to configuration file', required=True)
    args = parser.parse_args()

    config = cfg.load(args.config)
    window_sizes, model = models.waveunet(config['output_size'], 2, 2, config['down_kernel_size'], config['up_kernel_size'], config['depth'], config['num_filters'])
    print(window_sizes)

    out_dir, pred_dir, log_dir = output_dirs(config)
    cfg.save(out_dir + '/config.yml', config)
    print('Saving output to directory {}'.format(out_dir))

    device = torch.device(config['device'])
    model = model.to(device)

    train_names, val_names = training_fnames(config)
    if 'to_predict_path' in config.keys():
        to_predict_names = musdb18_basenames(config['to_predict_path'])
    else:
        to_predict_names = val_names

    with Report(log_dir) as report:
        for round in range(2):
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate_{}'.format(round)])
            for epoch in range(config['training_epochs']):
                random.shuffle(train_names)
                train(report, optimizer, config['batches_report'], epoch, device, model, audio_snippets_loader(round, config, window_sizes, train_names))

                if (epoch+1)==config['training_epochs'] or (epoch+1) % config['validation_epochs_frequency'] == 0:
                    eval_loss = evaluate(epoch, device, model, audio_snippets_loader(round, config, window_sizes, val_names))
                    report.add_scalar('eval_mse', eval_loss)
                    save_checkpoint(out_dir+'/checkpoint.pt', epoch, model, optimizer, eval_loss)

    print('Epoch: {:4d}\tApplying model to {} files.'.format(epoch, len(to_predict_names)))
    musdb18_transform(config['sampling_rate'], window_sizes, device, model, pred_dir, to_predict_names)

class Report():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()

    def advance_step(self):
        self.step += 1

    def add_scalar(self, tag, obj):
        self.writer.add_scalar(tag, obj, self.step)

if __name__ == '__main__':
    main()
