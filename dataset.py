import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import random
import pack_util

def init_types_and_id(root_dir):
    types = os.listdir(root_dir)
    types = [t for t in types if not t.endswith('.list')]
    types = sorted(types) #按字母顺序排序
    type2id = {}
    id2type = {}
    for i, t in enumerate(types):
        type2id[t] = i
        id2type[i] = t
    type2id['Other'] = len(types)
    id2type[len(types)] = 'Other'
    return type2id, id2type

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.type2id, self.id2type = init_types_and_id(root_dir)
        # print(self.type2id)
        '''
        self.type2id的内容如下，按字母顺序排序
        11类：{'Breath': 0, 'Brushing teeth': 1, 'Clapping': 2, 'Coughing': 3, 'Door knock': 4, 'Footsteps': 5, 'Keyboard typing': 6, 'Laughing': 7, 'Sneezing': 8, 'Snoring': 9, 'Other': 10}
         6类：{'Breath': 0, 'Brushing teeth': 1, 'Clapping': 2, 'Coughing': 3, 'Door knock': 4, 'Keyboard typing': 5, 'Other': 6}
        '''

        self.root_dir = root_dir
        with open(os.path.join(root_dir, 'train_set.list')) as fp:
            self.waves = fp.read().splitlines()

        self.waves2 = {}
        for wave in self.waves:
            type_name = os.path.split(wave)[0]
            if type_name not in self.waves2.keys():
                self.waves2[type_name] = []
            self.waves2[type_name].append(wave)
        # print(self.waves2)

        n_other = int(len(self.waves) * 0.15) #让各个类别比例相似
        self.waves += ['Other/666'] * n_other #后面的666不会被使用，随便写的

        noise_list = ['/local/data/wind-noise-record.lmdb']
        _, self.noise = pack_util.get_total(noise_list, keep_struct=False)
        self.sess_dict = pack_util.init_session(noise_list)

        speech_list = ['/local/data/from_zx/AEC/train_data/clean/clean-1000_vad_processed.lmdb']
        speech_list += ['/local/data/from_zx/AEC/train_data/clean/read_speech_500remain.lmdb']
        _, self.speech = pack_util.get_total(speech_list, keep_struct=False)
        self.sess_dict.update(pack_util.init_session(speech_list))

        self.other = self.noise + self.speech
    
    def __getitem__(self, idx):
        wave = self.waves[idx]
        type_name = os.path.split(wave)[0]
        type_id = self.type2id[type_name]
        if type_name != 'Other':
            wave_path = os.path.join(self.root_dir, wave)
            pcm, sr = sf.read(wave_path)
            pcm_16k = librosa.resample(pcm, sr, 16000)

            if random.uniform(0, 1) <= 0.4:
                same_type_waves = self.waves2[type_name]
                wave2 = random.choice(same_type_waves)
                wave2_path = os.path.join(self.root_dir, wave2)
                pcm2, sr = sf.read(wave2_path)
                pcm2_16k = librosa.resample(pcm2, sr, 16000)
                beg = int(random.uniform(0, 0.05) * pcm_16k.shape[0])
                length = int(random.uniform(0.4, 0.7) * pcm_16k.shape[0])
                pcm1 = pcm_16k[beg:beg+length]
                pcm2 = pcm2_16k[:pcm_16k.shape[0] - length]
                # print(beg, pcm1.shape[0] + pcm2.shape[0])
                pcm_16k = np.concatenate((pcm1, pcm2), axis=0)
                # sf.write('./test.wav', pcm_16k, 16000)

            if random.uniform(0, 1) <= 0.5: #均匀分布，50%情况加噪声
                idx_noise = random.randint(0, len(self.noise) - 1)
                noise = pack_util.load_audio(self.noise[idx_noise], self.sess_dict, 5)
                snr = np.random.choice([3, 5, 7, 9, 11, 15])
                # sf.write(f'./pcm_16k{snr}.wav', pcm_16k, 16000)
                pcm_16k, _, _ = pack_util.gen_mix(pcm_16k, noise, snr, energy_norm=False)
                # sf.write(f'./mix{snr}.wav', pcm_16k, 16000)
        else: #这里加入一些不相关数据，有说话声和纯噪声
            idx_other = random.randint(0, len(self.other) - 1)
            pcm_16k = pack_util.load_audio(self.other[idx_other], self.sess_dict, 5)
            # sf.write('./other.wav', pcm_16k, 16000)

        spec = librosa.stft(pcm_16k, n_fft=1024, hop_length=512, window='hann', center=False) #center=False, 不进行padding
        spec = spec[1:, :].T #不使用0频率
        spec_mag = np.abs(spec) #频域能量，不使用相位信息
        # print(spec_mag.shape, spec_mag[10, 0:10])
        # sf.write(f'./type{type_id}.wav', pcm_16k, 16000)
        return spec_mag, type_id

    def __len__(self):
        return len(self.waves)

class DevDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.types2id, self.id2types = init_types_and_id(root_dir)
        self.root_dir = root_dir
        with open(os.path.join(root_dir, 'dev_set.list')) as fp:
            self.waves = fp.read().splitlines()
    
    def __getitem__(self, idx):
        wave = self.waves[idx]
        type_name = os.path.split(wave)[0]
        type_id = self.types2id[type_name]
        if type_name != 'Other':
            wave_path = os.path.join(self.root_dir, wave)
            pcm, sr = sf.read(wave_path)
            pcm_16k = librosa.resample(pcm, sr, 16000)

        else: #这里加入一些不相关数据，有说话声和纯噪声
            raise NotImplementedError()

        spec = librosa.stft(pcm_16k, n_fft=1024, hop_length=512, window='hann', center=False)
        spec = spec[1:, :].T #不使用0频率
        spec_mag = np.abs(spec) #频域能量，不使用相位信息
        return spec_mag, type_id

    def __len__(self):
        return len(self.waves)


if __name__ == '__main__':
    dataset = TrainDataset('/local/data/zcs/sound_set')
    # dataset = DevDataset('/local/data/zcs/sound_set')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0)
    print(len(dataset))
    for idx, (feature, t) in enumerate(dataloader):
        print(idx, feature.shape, t)
        if idx >= 10:
            break
