import os
import io
import numpy as np
import lmdb
import soundfile
import librosa
import random
import zlib

_ZLIB_H = [bytes.fromhex(h) for h in ['7801', '785E', '789C', '78DA']]

def get_total(source_list, keep_struct=False):
    total = 0
    all_samples = []

    for source in source_list:
        if not os.path.exists(os.path.expanduser(source)):
            raise ValueError('data source file is not exists: {}'.format(source))

        extname = os.path.splitext(source)[-1].strip('.').split('.')[-1]
        if extname in ['lmdb']:
            env = lmdb.open(os.path.expanduser(source),
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            map_size=int(2**40))
            n = env.stat()['entries']
            total += n
            samples = [['lmdb', source, i] for i in range(n)]
            all_samples += [samples] if keep_struct else samples
            env.close()
        elif extname in ['flac', 'wav', 'pcm']:
            total += 1
            samples = [['file', os.path.expanduser(source)]]
            all_samples += [samples] if keep_struct else samples
        else:
            raise NotImplementedError('not implemented: {}'.format(source))

    return total, all_samples

def init_session(source_list):
    sess_dict = {}

    for source in source_list:
        if not os.path.exists(os.path.expanduser(source)):
            raise ValueError('data source file is not exists: {}'.format(source))

        extname = os.path.splitext(source)[-1].strip('.').split('.')[-1]
        if extname in ['lmdb']:
            env = lmdb.open(os.path.expanduser(source),
                            readonly=True,
                            lock=False,
                            readahead=True,
                            map_size=int(2**30))
            txn = env.begin(write=False)
            sess_dict.update({source:txn})
        # else:
        #     raise NotImplementedError('not implemented: {}'.format(source))
    return sess_dict

def load_from_soundfile(sf, seconds=-1):
    frames = seconds * sf.samplerate if seconds > 0 else -1
    if frames >= sf.frames or frames == -1:
        data = sf.read(frames)
    else:
        offset = random.randint(0, sf.frames-frames-1)
        # print(offset, frames, sf.frames)
        try:
            sf.seek(offset)
            data = sf.read(frames)
        except Exception as e:
            print('Ignore exception in load_from_soundfile():', sf, e)
            data = np.zeros(frames, dtype=np.float32)
    return data, sf.samplerate

def load_from_lmdb(sess, idx, seconds=-1):
    try:
        data = sess.get('{}'.format(idx).encode())
        if data[:2] in _ZLIB_H:
            data = zlib.decompress(data)
        buff = io.BytesIO(data)
        with soundfile.SoundFile(buff) as sf:
            data, sr = load_from_soundfile(sf, seconds)
        return data, sr
    except Exception as e:
        raise e

def load_audio(sample, sess_dict, seconds=-1, pad=True, samplerate=16000):
    '''
    sample: list of length 3, the content is ['lmdb', 'lmdb_file_name', index]
    sess_dict: dict of lmdb env, eg: {'lmdb_file_name1':env1, 'lmdb_file_name2':env2, ...}; we read FLAC from lmdb env
    secondes: time to read
    pad: 如果FLAC长度不足senconds，补零
    samplerate: 如果FLAC解码后不是目标采样率，则重采样
    '''
    if sample[0] == 'lmdb':
        _, sess_key, idx = sample
        sess = sess_dict[sess_key]
        data, sr = load_from_lmdb(sess, idx, seconds)
    elif sample[0] == 'file':
        filename = sample[1]
        extname = os.path.splitext(filename)[-1]
        if extname.lower() in ['.wav', '.flac']:
            f = soundfile.SoundFile(filename)
        elif extname.lower() in ['.pcm', '.raw']:
            f = soundfile.SoundFile(filename,
                            format='RAW',
                            channels=1,
                            samplerate=samplerate,
                            subtype='PCM_16')
        else:
            raise ValueError(f'{sample} is not supported!')
        data, sr = load_from_soundfile(f, seconds)
    else:
        raise NotImplementedError('not implemented: {}'.format(sample[0]))

    if sr != samplerate:
        data = librosa.resample(data, sr, samplerate)

    if seconds != -1:
        cur_len = data.shape[0]
        if cur_len < seconds * samplerate:
            data = np.pad(data, [0, seconds*samplerate-cur_len], mode='constant')
    return data

def gen_mix(clean, noise, snr, energy_norm=True):
    if (np.sum(noise**2) * (10.0**(snr / 10.0))) == 0:
        alpha = 1
    else:
        alpha = np.sqrt(np.sum(clean**2) / (np.sum(noise**2) * (10.0**(snr / 10.0))))

    alpha = np.where(np.isnan(alpha), np.zeros_like(alpha), alpha)
    noise = noise * alpha
    mic = clean + noise

    if energy_norm:
        CONSTANT = 0.95  #energy nomalization
        c = np.sqrt(CONSTANT * mic.size / np.sum(mic**2))
    else:
        peak = np.max(np.abs(mic))
        if peak > 4:
            mic = np.clip(mic, -4, 4)
            c = 0.25
        elif peak > 1:
            c = 1 / peak
        else:
            c = 1

    return mic * c, clean * c, noise * c