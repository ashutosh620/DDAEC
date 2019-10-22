# -*- coding: utf-8 -*-
import random
import pickle

import numpy as np
import scipy.signal

from datasets import TrainingDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
import sys
import librosa
import torch

def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False) 
    for j, seq_len in enumerate(nframes):
        loss_mask.data[j, :, 0:seq_len] += 1.0
    return loss_mask

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num
    

def logging(log_path, log_name, checkpoint, eval_steps):
    if checkpoint.start_iter+1 == eval_steps and checkpoint.start_epoch == 0:
        with open(log_path+log_name, 'w') as f_log:
            f_log.write('epoch, step, train_loss, eval_loss\n')
            f_log.write('%d, %d, %.4f, %.4f\n' % (checkpoint.start_epoch, checkpoint.start_iter,
                                              checkpoint.train_loss, checkpoint.eval_loss))
    else:
        with open(log_path+log_name, 'a') as f_log:
            f_log.write('%d, %d, %.4f, %.4f\n' % (checkpoint.start_epoch, checkpoint.start_iter,
                                              checkpoint.train_loss,checkpoint.eval_loss))

def metric_logging(log_path, log_name, ind, lst):
    if ind == 0:
        with open(log_path+log_name, 'w') as f_log:
            str1 = 'EPOCH, STOI, SNR, PESQ\n'
            f_log.write(str1 + '{}, {:.4f}, {:.4f}, {:.4f}\n'.format(ind, lst[0], lst[1], lst[2]))
    else:
        with open(log_path+log_name, 'a') as f_log:
            f_log.write('{}, {:.4f}, {:.4f}, {:.4f}\n'.format(ind, lst[0], lst[1], lst[2]))
        
def plotting(fig_path, fig_name, label, output):
    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path+fig_name)
    gs = gridspec.GridSpec(2,1)
    
    ax1 = plt.subplot(gs[0])
    ax1.imshow(np.log10(label.T)/np.log10(np.max(label)+1e-7), origin='lower')
    ax1.set_title('label')
    
    ax2 = plt.subplot(gs[1])
    ax2.imshow(np.log10(output.T)/np.log10(np.max(label)+1e-7),origin='lower')
    ax2.set_title('est')
    
    pdf.savefig()
    pdf.close()
