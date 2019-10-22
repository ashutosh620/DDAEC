# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import timeit
import sys
class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""
    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift
    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones
class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r
    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels//self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels//self.r, H, -1))
        return out
        
class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil-1)*(self.twidth-1)-1
            setattr(self, 'pad{}'.format(i+1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i+1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size, dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i+1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i+1), nn.PReLU(self.in_channels))
    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i+1))(skip)
            out = getattr(self, 'conv{}'.format(i+1))(out)
            out = getattr(self, 'norm{}'.format(i+1))(out)
            out = getattr(self, 'prelu{}'.format(i+1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out
        
class Net(nn.Module):
    def __init__(self, device, L=512, width=64):
        super(Net, self).__init__()
        self.L = L
        self.frame_shift = self.L // 2
        self.N = 256
        self.B = 256
        self.H = 512
        self.P = 3
        self.device = device
        self.in_channels = 1
        self.out_channels = 1
        self.kernel_size = (2, 3)
        self.elu = nn.SELU(inplace=True)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.width = width
        
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(512)
        self.inp_prelu = nn.PReLU(self.width)
        
        self.enc_dense1 = DenseBlock(512, 5, self.width)
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm1 = nn.LayerNorm(256)
        self.enc_prelu1 = nn.PReLU(self.width)
        
        self.enc_dense2 = DenseBlock(256, 5, self.width)
        self.enc_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm2 = nn.LayerNorm(128)
        self.enc_prelu2 = nn.PReLU(self.width)
        
        self.enc_dense3 = DenseBlock(128, 5, self.width)
        self.enc_conv3 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm3 = nn.LayerNorm(64)
        self.enc_prelu3 = nn.PReLU(self.width)
        
        self.enc_dense4 = DenseBlock(64, 5, self.width)
        self.enc_conv4 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm4 = nn.LayerNorm(32)
        self.enc_prelu4 = nn.PReLU(self.width)
        
        self.enc_dense5 = DenseBlock(32, 5, self.width)
        self.enc_conv5 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm5 = nn.LayerNorm(16)
        self.enc_prelu5 = nn.PReLU(self.width)
        
        self.enc_dense6 = DenseBlock(16, 5, self.width)
        self.enc_conv6 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm6 = nn.LayerNorm(8)
        self.enc_prelu6 = nn.PReLU(self.width)
        
        
        
        self.dec_dense6 =  DenseBlock(8, 5, self.width)
        self.dec_conv6 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm6 = nn.LayerNorm(16)
        self.dec_prelu6 = nn.PReLU(self.width)
        
        self.dec_dense5 =  DenseBlock(16, 5, self.width)
        self.dec_conv5 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm5 = nn.LayerNorm(32)
        self.dec_prelu5 = nn.PReLU(self.width)
        
        self.dec_dense4 =  DenseBlock(32, 5, self.width)
        self.dec_conv4 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm4 = nn.LayerNorm(64)
        self.dec_prelu4 = nn.PReLU(self.width)
        
        self.dec_dense3 =  DenseBlock(64, 5, self.width)
        self.dec_conv3 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm3 = nn.LayerNorm(128)
        self.dec_prelu3 = nn.PReLU(self.width)
        
        self.dec_dense2 =  DenseBlock(128, 5, self.width)
        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 =nn.LayerNorm(256)
        self.dec_prelu2 = nn.PReLU(self.width)
        
        self.dec_dense1 =  DenseBlock(256, 5, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(512)
        self.dec_prelu1 = nn.PReLU(self.width)
        
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))
        self.ola = TorchOLA(self.frame_shift)
        
    def forward(self, x):
        
        enc_list = []
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))
        
        out = self.enc_dense1(out)
        out = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense2(out)
        out = self.enc_prelu2(self.enc_norm2(self.enc_conv2(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense3(out)
        out = self.enc_prelu3(self.enc_norm3(self.enc_conv3(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense4(out)
        out = self.enc_prelu4(self.enc_norm4(self.enc_conv4(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense5(out)
        out = self.enc_prelu5(self.enc_norm5(self.enc_conv5(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense6(out)
        out = self.enc_prelu6(self.enc_norm6(self.enc_conv6(self.pad1(out))))
        enc_list.append(out)
        
        out = self.dec_dense6(out)
        out = torch.cat([out, enc_list[-1]], dim=1)
        out = self.dec_prelu6(self.dec_norm6(self.dec_conv6(self.pad1(out))))
        
        out = self.dec_dense5(out)
        out = torch.cat([out, enc_list[-2]], dim=1)
        out = self.dec_prelu5(self.dec_norm5(self.dec_conv5(self.pad1(out))))
        
        out = self.dec_dense4(out)
        out = torch.cat([out, enc_list[-3]], dim=1)
        out = self.dec_prelu4(self.dec_norm4(self.dec_conv4(self.pad1(out))))
        
        out = self.dec_dense3(out)
        out = torch.cat([out, enc_list[-4]], dim=1)
        out = self.dec_prelu3(self.dec_norm3(self.dec_conv3(self.pad1(out))))
        
        out = self.dec_dense2(out)
        out = torch.cat([out, enc_list[-5]], dim=1)
        out = self.dec_prelu2(self.dec_norm2(self.dec_conv2(self.pad1(out))))
        
        out = self.dec_dense1(out)
        out = torch.cat([out, enc_list[-6]], dim=1)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(out))))
        
        out = self.out_conv(out)
        out = self.ola(out)
        
        return out
        
        
        
