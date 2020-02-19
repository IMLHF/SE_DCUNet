import argparse
import os

import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from scipy.io import wavfile
import librosa
from tqdm import tqdm

import utils
from models.unet import Unet
from models.layers.istft import ISTFT
from se_dataset import AudioDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='exp/unet16.json', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
args = parser.parse_args()

n_fft, hop_length = 1024, 256
window = torch.hann_window(n_fft).cuda()
def stft(x):
    return torch.stft(x, n_fft, hop_length, window=window)
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()


def main():
    json_path = os.path.join(args.model_dir)
    params = utils.Params(json_path)

    net = Unet(params.model).cuda()
    # TODO - check exists
    checkpoint = torch.load('./final.pth.tar')
    net.load_state_dict(checkpoint)

    # train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='val')
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
    #                                collate_fn=train_dataset.collate, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1,
                                  collate_fn=test_dataset.collate, shuffle=False, num_workers=0)

    torch.set_printoptions(precision=10, profile="full")


    train_bar = tqdm(test_data_loader, ncols=60)
    cnt = 1
    with torch.no_grad():
        for input_ in train_bar:
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input_)
            mixed = stft(train_mixed).unsqueeze(dim=1)
            real, imag = mixed[..., 0], mixed[..., 1]
            out_real, out_imag = net(real, imag)
            out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
            out_audio = istft(out_real, out_imag, train_mixed.size(1))
            out_audio = torch.squeeze(out_audio, dim=1)
            for i, l in enumerate(seq_len):
                out_audio[i, l:] = 0
            # librosa.output.write_wav('mixed.wav', train_mixed[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
            # librosa.output.write_wav('clean.wav', train_clean[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
            sf.write('enhanced_testset/enhanced_%03d.wav' % cnt, np.array(out_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], dtype=np.float32), 16000,)
            cnt += 1



if __name__ == '__main__':
    main()
