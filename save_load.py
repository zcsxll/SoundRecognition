import os
import torch

def find_last_checkpoint(checkpoint_dir):
    epochs = []
    for name in os.listdir(checkpoint_dir):
        if os.path.splitext(name)[-1] == '.pth':
            epochs += [int(name.strip('ckpt_epoch_.pth'))]
    if len(epochs) == 0:
        raise IOError('no checkpoint found in {}'.format(checkpoint_dir))
    return max(epochs)

def save_checkpoint(checkpoint_dir, epoch, model, optimizer=None, scheduler=None):
    checkpoint = {}
    checkpoint['epoch'] = epoch

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint['model'] = model_state_dict

    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()
        # for k, v in optimizer_state_dict.items():
        #     print(k, type(v))
        # optimizer_state_dict = rename_dict_key(optimizer_state_dict)
        checkpoint['optimizer'] = optimizer_state_dict
    else:
        checkpoint['optimizer'] = None

    if scheduler is not None:
        scheduler_state_dict = scheduler.state_dict()
        # for k, v in scheduler_state_dict.items():
        #     print(k, v)
        checkpoint['scheduler'] = scheduler_state_dict
    else:
        checkpoint['scheduler'] = None

    if not os.path.exists(checkpoint_dir):
        print(f'create dir [{checkpoint_dir}]')
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'ckpt_epoch_%02d.pth'% epoch))

def load_checkpoint(checkpoint_dir, epoch=-1):
    if epoch == -1:
        epoch = find_last_checkpoint(checkpoint_dir)
    checkpoint_name = 'ckpt_epoch_%02d.pth'% epoch
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return ckpt, checkpoint_path

def save_model(checkpoint_dir, epoch, model):
    save_checkpoint(checkpoint_dir, epoch, model, optimizer=None)

def load_model(checkpoint_dir, epoch, model):
    try:
        ckpt, checkpoint_path = load_checkpoint(checkpoint_dir, epoch)
        model_state_dict = ckpt['model']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        # elif isinstance(model, torchDDP):
        #     model.module.load_state_dict(model_state_dict)
        # elif isinstance(model, apexDDP):
        #     model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    except Exception as e:
        print('failed to load model, {}'.format(e))
        return model, None
    return model, checkpoint_path

def load_optimizer(checkpoint_dir, epoch, optimizer):
    try:
        ckpt, checkpoint_path = load_checkpoint(checkpoint_dir, epoch)
        optimizer_state_dict = ckpt['optimizer']
        optimizer.load_state_dict(optimizer_state_dict)
    except Exception as e:
        print('failed to load optimizer, {}'.format(e))
        return optimizer, None
    return optimizer, checkpoint_path

def load_scheduler(checkpoint_dir, epoch, scheduler):
    try:
        ckpt, checkpoint_path = load_checkpoint(checkpoint_dir, epoch)
        scheduler_state_dict = ckpt['scheduler']
        scheduler.load_state_dict(scheduler_state_dict)
    except Exception as e:
        print('failed to load scheduler, {}'.format(e))
        return scheduler, None
    return scheduler, checkpoint_path
