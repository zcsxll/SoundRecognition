import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import random
import pack_util

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        types = os.listdir(root_dir)
        types = [t for t in types if not t.endswith('.list')]
        types = sorted(types) #按字母顺序排序
        self.types2id = {}
        self.id2types = {}
        for i, t in enumerate(types):
            self.types2id[t] = i
            self.id2types[i] = t
        self.types2id['other'] = len(types)
        self.id2types[len(types)] = 'other'
        '''
        self.types2id的内容如下，按字母顺序排序
        {'Breath': 0, 'Brushing teeth': 1, 'Clapping': 2, 'Coughing': 3, 'Door knock': 4, 'Drinking, sipping': 5, 'Footsteps': 6, 'Keyboard typing': 7, 'Laughing': 8, 'Sneezing': 9, 'Snoring': 10, 'other': 11}
        '''

        self.root_dir = root_dir
        with open(os.path.join(root_dir, 'train_set.list')) as fp:
            self.waves = fp.read().splitlines()

        n_other = int(len(self.waves) * 0.1)
        self.waves += ['other/666'] * n_other #后面的666不会被使用，随便写的

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
        type_id = self.types2id[type_name]
        if type_name != 'other':
            wave_path = os.path.join(self.root_dir, wave)
            pcm, sr = sf.read(wave_path)
            pcm_16k = librosa.resample(pcm, sr, 16000)

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

        spec = librosa.stft(pcm_16k, n_fft=1024, hop_length=512, window='hann')
        spec = spec[1:, :].T #不使用0频率
        spec_mag = np.abs(spec) #频域能量，不使用相位信息
        # print(spec_mag.shape, spec_mag[10, 0:10])
        # sf.write(f'./type{type_id}.wav', pcm_16k, 16000)
        return spec_mag, type_id #spec_mag的shape是[431, 512]

    def __len__(self):
        return len(self.waves)

if __name__ == '__main__':
    dataset = TrainDataset('/local/data/zcs/sound_set')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0)
    # print(len(dataset))
    for idx, (feature, t) in enumerate(dataloader):
        print(idx, feature.shape, t)
        if idx >= 50:
            break