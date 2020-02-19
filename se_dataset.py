import numpy as np
from tqdm import tqdm
import librosa
import os
import torch
from torch.utils import data
import soundfile as sf


def repeat_to_len_2(wave1, wave2, repeat_len, random_trunc_long_wav=False):
    wave_len1 = len(wave1)
    wave_len2 = len(wave2)
    assert wave_len1 == wave_len2, 'wav length not match.'
    wave_len = wave_len1
    if random_trunc_long_wav and wave_len > repeat_len:
        random_s = np.random.randint(wave_len-repeat_len+1)
        wave1 = wave1[random_s:random_s+repeat_len]
        wave2 = wave2[random_s:random_s+repeat_len]
        return wave1, wave2

    while len(wave1) < repeat_len:
        wave1 = np.tile(wave1, 2)
        wave2 = np.tile(wave2, 2)
    wave1 = wave1[0:repeat_len]
    wave2 = wave2[0:repeat_len]
    return wave1, wave2

# Reference
# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder='./dataset', setname='train'):
    assert(setname in ['train', 'val'])

    dataset = {}
    foldername = folder + '/' + setname + 'set'

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []
    dataset['setname'] = setname

    filelist = os.listdir("%s_noisy" % (foldername))
    filelist = [f for f in filelist if f.endswith(".wav")]
    filelist.sort()
    for i in tqdm(filelist):
        dataset['innames'].append("%s_noisy/%s" % (foldername, i))
        dataset['outnames'].append("%s_clean/%s" % (foldername, i))
        dataset['shortnames'].append("%s" % i)

    return dataset

# DATA LOADING - LOAD FILE DATA
def load_data(dataset):

    dataset['inaudio'] = [None]*len(dataset['innames'])
    dataset['outaudio'] = [None]*len(dataset['outnames'])

    for id in tqdm(range(len(dataset['innames']))):

        if dataset['inaudio'][id] is None:
            inputData, sr = sf.read(dataset['innames'][id])
            outputData, sr = sf.read(dataset['outnames'][id])
            if dataset['setname'] == 'train':
                inputData, outputData = repeat_to_len_2(inputData, outputData, 16000*3, True)

            dataset['inaudio'][id] = np.float32(inputData)
            dataset['outaudio'][id] = np.float32(outputData)

    return dataset

class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):
        dataset = load_data_list(setname=data_type)
        self.dataset = load_data(dataset)

        self.file_names = dataset['innames']

    def __getitem__(self, idx):
        mixed = torch.from_numpy(self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        clean = torch.from_numpy(self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e] = inp # no padding
        return input_mat

    def collate(self, inputs):
        mixeds, cleans = zip(*inputs)
        seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

        x = torch.FloatTensor(self.zero_pad_concat(mixeds))
        y = torch.FloatTensor(self.zero_pad_concat(cleans))

        batch = [x, y, seq_lens]
        return batch
