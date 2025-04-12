import torch

def save_states(fname, model, model_d, optimizer, optimizer_d, 
                n_iter, epoch, net_config, config):
    torch.save({'model': model.state_dict(),
                'model_d': model_d.state_dict(),                
                'optim': optimizer.state_dict(),
                'optim_d': optimizer_d.state_dict(),
                'epoch': epoch, 'iter': n_iter,
                'net_config': net_config,
                },
               f'{config.checkpoint_dir}/{fname}')