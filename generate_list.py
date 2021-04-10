import os
import sys
import random

def generate(root_dir, random_seed=1234, dev_cnt=1):
    random.seed(random_seed)
    types = os.listdir(root_dir)
    types = [t for t in types if not t.endswith('.list')]

    train_set = []
    dev_set = []
    for t in types:
        type_dir = os.path.join(root_dir, t)
        waves = os.listdir(type_dir)
        waves = [wave for wave in waves if wave.endswith('.wav')]
        for i in range(1, 6): #每个文件的第一个字符都是数字1到5中的一个
            subset = [wave for wave in waves if wave.startswith(f'{i}')]
            assert dev_cnt < len(subset) // 2
            random.shuffle(subset)
            subset = [os.path.join(t, wave) for wave in subset]
            dev_set += subset[:dev_cnt]
            train_set += subset[dev_cnt:]

    with open(os.path.join(root_dir, 'train_set.list'), 'w') as fp:
        for train in train_set:
            fp.write(train + '\n')
    with open(os.path.join(root_dir, 'dev_set.list'), 'w') as fp:
        for dev in dev_set:
            fp.write(dev + '\n')
    
if __name__ == '__main__':
    root_dir = '/local/data/zcs/sound_set'
    generate(root_dir)