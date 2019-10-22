# -*- coding: utf-8 -*-
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys
import random
class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        return torch.from_numpy(x).float()
class SetContext:
    r"""Concatenates consecutive frames to set context"""
    def __init__(self, feat_side=0):
        self.feat_side = feat_side
    def __call__(self, feat):
        if self.feat_side == 0:
            return feat
        assert len(feat.shape) >= 2
        feat_dim = feat.shape[-1]
        nframes = feat.shape[-2]
        paddings = np.zeros(list(feat.shape[:-2]) + [self.feat_side, feat_dim])
        feat_padded = np.concatenate([paddings, feat, paddings], axis=-2)
        f_list = []
        for offset in range(2*self.feat_side+1):
            f_list.append(feat_padded[..., offset:(nframes+offset), :])
        ans = np.concatenate(f_list, axis=-1)
        return ans
class AverageContext:
    r"""Averages multiple predictions of a frame in different contexts"""
    def __init__(self, dimension=257, order=2):
        self.dimension=dimension
        self.order = order
    def __call__(self, inputs):
        if self.order == 0:
            return inputs
        m = 2*self.order+1
        witdth = m*self.dimension
        inds = []
        beg = 0
        end = beg + self.dimension
        for i in range(m):
            inds.append(range(beg, end))
            beg += self.dimension
            end += self.dimension
        inds = np.reshape(inds, (m, -1)).astype(np.int32)
        N = inputs.shape[-2]
        ans = np.zeros(inputs.shape[:-1] + (self.dimension, ))

        for i in range(N):
            cnt = 1
            temp = inputs[..., i, inds[self.order]]
            for j in range(1, self.order + 1):
                if (i - j) >= 0:
                    temp = temp + inputs[..., i - j, inds[self.order + j]]
                    cnt = cnt + 1
                if  (i + j) < N:
                    temp = temp + inputs[..., i + j, inds[self.order - j]]
                    cnt = cnt + 1
            ans[..., i, :] = temp / cnt
        return ans
            
class OLA:
    r"""Performs overlap-and-add""" 
    def __init__(self, frame_shift=256):
        self.frame_shift = frame_shift
    def __call__(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones
class SignalToFrames:
    r"""Chunks a signal into frames"""
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
    def __call__(self, in_sig):
        frame_size = self.frame_size
        frame_shift = self.frame_shift
        sig_len = in_sig.shape[-1]
        nframes = (sig_len // self.frame_shift) 
        a = np.zeros(list(in_sig.shape[:-1]) + [nframes, self.frame_size])
        start = 0
        end = start + self.frame_size
        k=0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size]=in_sig[..., start:]
                
            start = start + self.frame_shift
            end = start + self.frame_size
        return a
class STFT:
    r"""Computes STFT of a signal"""
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.get_frames = SignalToFrames(self.frame_size, self.frame_shift)
    def __call__(self, signal):
        frames = self.get_frames(signal)
        frames = frames*self.win
        feature = np.fft.fft(frames)[..., 0:(self.frame_size//2+1)]
        feat_R = np.real(feature)
        feat_I = np.imag(feature)
        feature = np.stack([feat_R, feat_I], axis=0)
        return feature
class ISTFT:
    r"""Computes inverse STFT"""
    # includes overlap-and-add
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.ola = OLA(self.frame_shift)
    def __call__(self, stft):
        R = stft[0:1, ...]
        I = stft[1:2, ...]
        cstft = R + 1j*I
        fullFFT = np.concatenate((cstft, np.conj(cstft[..., -2:0:-1])), axis=-1)
        T = np.fft.ifft(fullFFT)
        T = np.real(T)
        T = T / self.win
        signal = self.ola(T)
        return signal.astype(np.float32)
class TrainingDataset(Dataset):
    r"""Training dataset."""
    
    def __init__(self, file_list, frame_size=512, frame_shift=256, nsamples=64000):
        self.file_list = file_list
        self.nsamples = nsamples
        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()
    def __len__(self):
        print(len(self.file_list))
        return len(self.file_list)
        
    def __getitem__(self, index):
        filename = self.file_list[index]
        filename = filename
        reader = h5py.File(filename,'r')
        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]
        reader.close()
        
        size = feature.shape[0]
        start = random.randint(0, max(0, size+1-self.nsamples))
        feature = feature[start:start+self.nsamples]
        label = label[start:start+self.nsamples]
        
        feature = np.reshape(feature, [1, -1])
        label = np.reshape(label, [1, -1])
        
        feature = self.get_frames(feature)
        
        
        feature = self.to_tensor(feature)
        label = self.to_tensor(label)
        
        return feature, label
        
    
class EvalDataset(Dataset):
    r"""Evaluation dataset."""
    
    def __init__(self, filename, length):
        self.filename = filename
        self.length = length
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        reader = h5py.File(self.filename,'r')
        reader_grp = reader[str(index)]
        feature = reader_grp['noisy_raw'][:]
        label = reader_grp['clean_raw'][:]
        
        return feature, label
   
class Compression(object):
    r"""Root cubic root compression."""
    def __call__(self, feature):
        return feature**(1./3.)
            

class TrainCollate(object):
    
    def __init__(self):
        self.name = 'collate'
    
    def __call__(self, batch):
        if isinstance(batch,list):
            feat_dim = batch[0][0].shape[-1]
            label_dim = batch[0][1].shape[-1]
            
            
            feat_nchannels = batch[0][0].shape[0]
            label_nchannels = batch[0][1].shape[0]
            sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
            lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))
              
            padded_feature_batch = torch.zeros((len(lengths), feat_nchannels, lengths[0][0], feat_dim))
            padded_label_batch = torch.zeros((len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths), ), dtype=torch.int32)
            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0:lengths[i][0], :] = sorted_batch[i][0]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]
           
            return padded_feature_batch, padded_label_batch, lengths1
        else:
            raise TypeError('`batch` should be a list.')
class EvalCollate(object):
    
    def __init__(self):
        self.name = 'collate'
    
    def __call__(self, batch):
        if isinstance(batch,list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')
