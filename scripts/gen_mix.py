# -*- coding: utf-8 -*-

import random
import sys
import os
import time
from datetime import datetime

import gflags
import h5py
import soundfile as sf
import numpy as np
import pymp
from progressbar import ProgressBar, Bar, Timer, ETA, Percentage




########################### 1. Configurations
######### parse commands
FLAGS = gflags.FLAGS
gflags.DEFINE_string('mode','','train or test')
gflags.DEFINE_boolean('unseen',False,'is_unseen_test: True or False')
FLAGS(sys.argv)

######### file paths
filelists_path = '../filelists/'
speech_path = '../../premix_data/WSJ0_83spks/' # path to the utternaces of 83 speakers from WSJ0 SI-84
train_noise_path = '../../premix_data/noise_10000/' # directory of the 10000 training noises
test_noise_path = '../../premix_data/noise/' # directory of the test noises
test_mixture_path = '../data/mixture/test/' 
train_mixture_path = '../data/mixture/train/'

######### settings
num_train_sentences = 320000
num_workers = 6
folder_cap = 5000 # each folder includes 5000 mixture files at most

constant = 1.0 # used for energy normalization

train_snr_list = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
test_snr_list = [-5.0, -2.0, 0.0, 2.0, 5.0]

train_noise = 'long_wave.bin' # All the 10000 noises are concatenated and stored in a binary files
test_noise_list = ['ADTbabble.wav', 'ADTcafeteria.wav', 'factory.wav']
srate = 16000

###############################################
########################### 2. Generating mixtures
if FLAGS.mode == 'train':
    speech_list = open(filelists_path+'trainFileList.txt','r')
    filelines_speech = speech_list.readlines()
    speech_list.close()
    print('[%s] Using noise: %s' % (FLAGS.mode,train_noise))
    print('SNR level: %s dB' % train_snr_list)
    print('%d sentences in total' % num_train_sentences)
    # read noise
    n = np.memmap(train_noise_path+train_noise, dtype=np.float32, mode='r')
    
    # custom a progressbar
    widgets = ['[',Timer(),'](',ETA(),')',Bar('='),'(',Percentage(),')']
    bar = ProgressBar(widgets=widgets,max_value=num_train_sentences)
    bar_count = pymp.shared.list()
    bar_count.append(0)
    with pymp.Parallel(num_workers) as p:
        for count in p.range(num_train_sentences):
            # write all examples into an h5py file
            filefolder = '%d-%d/' % ((count//folder_cap)*folder_cap,(count//folder_cap+1)*folder_cap-1)
            filename = '%s_%d.samp' % (FLAGS.mode,count)
            if not os.path.isdir(train_mixture_path+filefolder):
                os.makedirs(train_mixture_path+filefolder)
            writer = h5py.File(train_mixture_path+filefolder+filename,'w')

            random.seed(datetime.now())
            s_c = random.randint(0,len(filelines_speech)-1)
            snr_c = random.randint(0,len(train_snr_list)-1)

            speech_name = filelines_speech[s_c].strip()
            s, srate_s = sf.read(speech_path+speech_name)
            if srate_s != 16000:
                raise ValueError('Invalid sample rate!')
            snr = train_snr_list[snr_c]

            # choose a point where we start to cut
            start_cut_point = random.randint(0,n.size-s.size)
            while np.sum(n[start_cut_point:start_cut_point+s.size]**2.0) == 0.0:
                start_cut_point = random.randint(0,n.size-s.size)
            # cut noise
            n_t = n[start_cut_point:start_cut_point+s.size]
           
            alpha = np.sqrt(np.sum(s**2.0)/(np.sum(n_t**2.0)*(10.0**(snr/10.0))))
            snr_check = 10.0*np.log10(np.sum(s**2.0)/(np.sum((n_t*alpha)**2.0)))
            mix = s + alpha * n_t
            # energy normalization
            c = np.sqrt(constant*mix.size/np.sum(mix**2.0))
            mix = mix * c
            s = s * c
            
            writer.create_dataset('noisy_raw', data=mix.astype(np.float32), chunks=True)
            writer.create_dataset('clean_raw', data=s.astype(np.float32), chunks=True)    
            writer.close()
            
            with p.lock:
                bar.update(bar_count[0])
                bar_count[0] += 1
                
    bar.finish()    
    print('sleep for 3 secs...')
    time.sleep(3)
    f_train_list = open(filelists_path+'train_list.txt','w')
    for count in range(num_train_sentences):
        filefolder = '%d-%d/' % ((count//folder_cap)*folder_cap,(count//folder_cap+1)*folder_cap-1)
        filename = '%s_%d.samp' % (FLAGS.mode,count)
        f_train_list.write(train_mixture_path+filefolder+filename+'\n')
    f_train_list.close()
elif FLAGS.mode == 'test':
    if not os.path.isdir(test_mixture_path):
        os.makedirs(test_mixture_path)
    if FLAGS.unseen == True:
        speech_list = open(filelists_path+'testFileList_unseen.txt','r')
        print('[%s] Using unseen speakers' % (FLAGS.mode))
    else:
        speech_list = open(filelists_path+'testFileList_seen.txt','r')
        print('[%s] Using seen speakers' % (FLAGS.mode))
    filelines_speech = speech_list.readlines()
    speech_list.close()
    for noise_name in test_noise_list:
        print('Using %s noise' % (noise_name))
        # read noise
        n, srate_n = sf.read(test_noise_path+noise_name)
        if len(n.shape) == 2:
            n = n[:, 0]
        if srate_n != 16000:
            raise ValueError('Invalid sample rate!')
        for snr in test_snr_list:
            print('SNR level: %d dB' % snr)
            # write all examples into h5py files
            if FLAGS.unseen == True:
                filename_mix = '%s_%s_snr%d_unseen_mix.dat' % (FLAGS.mode,noise_name.split('.')[0],snr)
                filename_s = '%s_%s_snr%d_unseen_s.dat' % (FLAGS.mode,noise_name.split('.')[0],snr)
                filename = '%s_%s_snr%d_unseen.samp' % (FLAGS.mode,noise_name.split('.')[0],snr)
            else:
                filename_mix = '%s_%s_snr%d_seen_mix.dat' % (FLAGS.mode,noise_name.split('.')[0],snr)
                filename_s = '%s_%s_snr%d_seen_s.dat' % (FLAGS.mode,noise_name.split('.')[0],snr)
                filename = '%s_%s_snr%d_seen.samp' % (FLAGS.mode,noise_name.split('.')[0],snr)
            f_mix = h5py.File(test_mixture_path+filename_mix,'w')
            f_s = h5py.File(test_mixture_path+filename_s,'w')
            writer = h5py.File(test_mixture_path+filename,'w')
            # custom a progressbar
            widgets = ['[',Timer(),'](',ETA(),')',Bar('='),'(',Percentage(),')']
            bar = ProgressBar(widgets=widgets)
            for i in bar(range(len(filelines_speech))):
                speech_name = filelines_speech[i].strip()
                s, srate_s = sf.read(speech_path+speech_name)
                if srate_s != 16000:
                    raise ValueError('Invalid sample rate!')
                # choose a point where we start to cut
                start_cut_point = random.randint(0, n.size-s.size)
                while np.sum(n[start_cut_point:start_cut_point+s.size]**2.0) == 0.0:
                    start_cut_point = random.randint(0, n.size-s.size)
                # cut noise
                n_t = n[start_cut_point:start_cut_point+s.size]
                # mixture = speech + noise
                alpha = np.sqrt(np.sum(s**2.0)/(np.sum(n_t**2.0)*(10.0**(snr/10.0))))
                snr_check = 10.0*np.log10(np.sum(s**2.0)/np.sum((n_t*alpha)**2.0))
                mix = s + alpha * n_t
                
                # energy normalization
                c = np.sqrt(constant*mix.size/np.sum(mix**2))
                mix = mix * c
                s = s * c
                
                f_mix.create_dataset(str(i), data=mix.astype(np.float32), chunks=True)
                f_s.create_dataset(str(i), data=s.astype(np.float32), chunks=True)
            
                
                writer_grp = writer.create_group(str(i))
                writer_grp.create_dataset('noisy_raw', data=mix.astype(np.float32), chunks=True)
                writer_grp.create_dataset('clean_raw', data=s.astype(np.float32), chunks=True)
            
            f_mix.close()
            f_s.close()
            writer.close()
            
            print('Written into %s' % (filename))

            print('sleep for 3 secs...')
            time.sleep(3)
else:
    raise ValueError('Invalid mode!')
        
print('[%s] Finish generating mixtures.\n' % FLAGS.mode)    
