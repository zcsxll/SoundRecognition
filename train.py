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

def train(model,
        loss_fn,
        optimizer,
        scheduler,
        train_dataloader,
        logger, epoch):
    model.train()

    pbar = tqdm(total=len(train_dataloader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    pbar.set_description(f'Epoch %d' % epoch)

    for step, (features, types) in enumerate(train_dataloader):
        features = features.cuda()
        types = types.cuda()
        preds = model(features)
        # print(preds, types)
        loss = loss_fn(preds, types)
        
        pbar.set_postfix(**{'loss':loss.detach().cpu().item()})
        pbar.update()

        optimizer.zero_grad()
        loss.backward()
        total_grad_norm = cg.clip_grad_norm(model.parameters(), 2)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            logger.add_scalar('train/loss', loss.cpu().detach().numpy(), scheduler.gstep)
            logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], scheduler.gstep)
    pbar.close()

def main(conf):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)

    logger = SummaryWriter(logdir=conf['checkpoint'])

    train_dataset = dataset.TrainDataset('/local/data/zcs/sound_set')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=16,
                                                shuffle=True,
                                                num_workers=4)
    
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
        # evaluate(model, eval_dataloader, evaluation_logger, epoch, conf)
        # break

if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])
    # main('./conf/LSTM_128_4.yaml')