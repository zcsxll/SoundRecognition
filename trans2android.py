import os
import sys
import yaml
import importlib
import torch

import save_load as sl

def import_class(class_str):
    class_str = class_str.split('.')
    module_name = '.'.join(class_str[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_str[-1])

def trans(conf, epoch):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)
 
    model = import_class(conf['model']['name'])(**conf['model']['args'])
    print('total model parameters: %.2f K' % (model.total_parameter() / 1024))
    
    model, checkpoint_path = sl.load_model(conf['checkpoint'], epoch, model)
    print(f'load from {checkpoint_path}')
    out_name = os.path.split(conf['checkpoint'])[-1] + '.pt'

    model.eval()
    example = torch.rand(1, 157, 512)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(out_name)

if __name__ == '__main__':
    print(torch.__version__)
    # assert len(sys.argv) == 3
    # main(sys.argv[1], int(sys.argv[2]))
    trans('./conf/CNN_16x3_32.yaml', 69)
