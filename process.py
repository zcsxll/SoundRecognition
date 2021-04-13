import os
import sys
import yaml
import importlib
import torch
import soundfile as sf
import librosa
import numpy as np

import save_load as sl

def import_class(class_str):
    class_str = class_str.split('.')
    module_name = '.'.join(class_str[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_str[-1])

def init_types_and_id(root_dir):
    types = os.listdir(root_dir)
    types = [t for t in types if not t.endswith('.list')]
    types = sorted(types) #按字母顺序排序
    types2id = {}
    id2types = {}
    for i, t in enumerate(types):
        types2id[t] = i
        id2types[i] = t
    types2id['Other'] = len(types)
    id2types[len(types)] = 'Other'
    return types2id, id2types

def process(conf, wave_path, epoch):
    types2id, id2types = init_types_and_id('/local/data/zcs/sound_set/')

    with open(conf) as fp:
        conf = yaml.safe_load(fp)

    model = import_class(conf['model']['name'])(**conf['model']['args'])
    print('total model parameters: %.2f K' % (model.total_parameter() / 1024))
    model, checkpoint_path = sl.load_model(conf['checkpoint'], int(epoch), model)
    print(f'load from {checkpoint_path}')

    model.eval()

    pcm, sr = sf.read(wave_path)
    spec = librosa.stft(pcm, n_fft=1024, hop_length=512, window='hann', center=False)
    spec_mag = np.abs(spec[1:, :].T) #频域能量，不使用相位信息
    tensor = torch.from_numpy(spec_mag).unsqueeze(0)

    preds = model(tensor)
    p = torch.nn.Softmax()(preds).detach().numpy()
    #for i, v in enumerate(preds.detach().numpy()):
    for i, v in enumerate(p):
        print(id2types[i], '\t%.2f' % v)

if __name__ == '__main__':
    assert len(sys.argv) == 4
    process(sys.argv[1], sys.argv[2], sys.argv[3])
    #process('./conf/CNN_16x3_32.yaml', './waves/2021-04-11__21-31-25.wav', 216)
