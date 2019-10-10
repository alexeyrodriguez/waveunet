import torch

import models

def create_waveunet_model(config):
    window_sizes, model = models.waveunet(config['output_size'], 2, 2, config['down_kernel_size'], config['up_kernel_size'], config['depth'], config['num_filters'])
    print('Creating WaveUNet model with input output sample size: {}'.format(window_sizes))
    return window_sizes, model

def save_checkpoint(outdir, epoch, model, optimizer, eval_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eval_loss': eval_loss
        }, outdir+'/checkpoint.pt')

def load_model(path, model):
    checkpoint = torch.load(path+'/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
