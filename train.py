import os
import sys
import yaml
import importlib
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import dataset
import noam
import save_load as sl
import clip_grad as cg

def import_class(class_str):
    class_str = class_str.split('.')
    module_name = '.'.join(class_str[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_str[-1])

def calc_correct(preds, types):
    preds = torch.nn.Softmax(dim=-1)(preds)
    preds = torch.max(preds, dim=-1).indices
    # acc = torch.mean((preds == types).float())
    correct = torch.sum((preds == types).float())
    return correct.detach().cpu().numpy()

def train(model,
        loss_fn,
        optimizer,
        scheduler,
        train_dataloader,
        logger, epoch):
    model.train()

    pbar = tqdm(total=len(train_dataloader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    pbar.set_description(f'Epoch %d' % epoch)

    correct = 0
    total = 0
    for step, (features, types) in enumerate(train_dataloader):
        total += types.shape[0]

        features = features.cuda()
        types = types.cuda()
        preds = model(features)
        # print(preds, types)
        correct += calc_correct(preds, types)
        loss = loss_fn(preds, types)
        
        pbar.set_postfix(**{'loss':loss.detach().cpu().item(), 'acc':correct/total})
        pbar.update()

        optimizer.zero_grad()
        loss.backward()
        cg.clip_grad_norm(model.parameters(), 1) #全部参数的梯度的标准差之和不可以大于1
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            logger.add_scalar('train/loss', loss.cpu().detach().numpy(), scheduler.gstep)
            logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], scheduler.gstep)
    logger.add_scalar('accuracy/train', correct/total, epoch)
    pbar.close()

@torch.no_grad()
def evaluate(model, dev_dataloader, logger, epoch):
    model.eval()

    correct = 0
    total = 0
    for step, (features, types) in enumerate(tqdm(dev_dataloader)):
        total += types.shape[0]

        features = features.cuda()
        types = types.cuda()
        preds = model(features)

        correct += calc_correct(preds, types)

    acc = correct / total
    logger.add_scalar('accuracy/dev', acc, epoch)
    print(f'dev accuracy: {acc}, {correct}, {total}')

def main(conf):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)

    logger = SummaryWriter(logdir=conf['checkpoint'])

    train_dataset = dataset.TrainDataset('/local/data/zcs/sound_set')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=16,
                                                shuffle=True,
                                                num_workers=4)

    dev_dataset = dataset.DevDataset('/local/data/zcs/sound_set')
    dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2)
    
    model = import_class(conf['model']['name'])(**conf['model']['args'])
    print('total model parameters: %.2f K' % (model.total_parameter() / 1024))
    os.environ["CUDA_VISIBLE_DEVICES"] = conf['gpu_ids']
    model = model.cuda()
    model, checkpoint_path = sl.load_model(conf['checkpoint'], -1, model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())
    optimizer, checkpoint_path = sl.load_optimizer(conf['checkpoint'], -1, optimizer)

    scheduler = noam.LRScheduler(optimizer, warmup_steps=1000, init_lr=0.001)
    scheduler, checkpoint_path = sl.load_scheduler(conf['checkpoint'], -1, scheduler)

    try:
        trained_epoch = sl.find_last_checkpoint(conf['checkpoint'])
        print('train form epoch %d' % (trained_epoch + 1))
    except Exception as e:
        print('train from the very begining, {}'.format(e))
        trained_epoch = -1
    for epoch in range(trained_epoch + 1, 1000):
        train(model, loss_fn, optimizer, scheduler, train_dataloader, logger, epoch)
        sl.save_checkpoint(conf['checkpoint'], epoch, model, optimizer, scheduler)
        evaluate(model, dev_dataloader, logger, epoch)
        # break

if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])
    # main('./conf/LSTM_128_4.yaml')