# -*- coding: utf-8 -*-

import os

import h5py
import numpy as np
import soundfile as sf
from progressbar import ProgressBar, Bar, Timer, ETA, Percentage, DynamicMessage

from STOI import stoi
from PESQ import getPESQ
import timeit
import librosa

def normalize_wav(sig):
    scale = np.max(np.abs(sig)) + 1e-7
    sig = sig / scale
    
    return sig, scale


def snr(s, s_p):
    r""" calculate signal-to-noise ratio (SNR)
    
        Parameters
        ----------
        s: clean speech
        s_p: processed speech
    """
    return 10.0 * np.log10(np.sum(s**2)/np.sum((s_p-s)**2))
    

class Metric(object):
    
    def __init__(self, args):
        with open(args.assess_list,'r') as assess_list_file:
            self.assess_list = [line.strip() for line in assess_list_file.readlines()]
        print('assess list:', self.assess_list)
        
        self.tool_path = args.tool_path
        self.wav_path = args.wav_path
        self.test_mixture_path = args.test_mixture_path
        self.prediction_path = args.prediction_path
        self.num_test_sentences = args.num_test_sentences
        self.srate = args.srate
        
        
        
    def getSTOI(self):
        for i in range(len(self.assess_list)):
            #print '\n[%d/%d] Assess on %s...' % ((i+1),len(self.assess_list), self.assess_list[i])
            start1 = timeit.default_timer()
            print('')
            print('{}/{}, Started assess on {} ...'.format(i+1, len(self.assess_list), self.assess_list[i]))
            print('')
            f_mix = h5py.File(os.path.join(self.test_mixture_path, self.assess_list[i]+'_mix.dat'), 'r')
            f_s = h5py.File(os.path.join(self.test_mixture_path, self.assess_list[i]+'_s.dat'), 'r')
            f_est_s = h5py.File(os.path.join(self.prediction_path, self.assess_list[i]+'_s_est.dat'), 'r')
            f_ideal_s = h5py.File(os.path.join(self.prediction_path, self.assess_list[i]+'_s_ideal.dat'), 'r')
    
            # create arrays for stoi and snr
            est_stoi_s_accu = 0.0
            ideal_stoi_s_accu = 0.0
            mix_stoi_s_accu = 0.0
            est_snr_s_accu = 0.0
            ideal_snr_s_accu = 0.0
            mix_snr_s_accu = 0.0
            
            for j in range(self.num_test_sentences):
                mix = f_mix[str(j)][:]
                s = f_s[str(j)][:]
                est_s = f_est_s[str(j)][:]
                ideal_s = f_ideal_s[str(j)][:]

                
                # compute stoi
                est_stoi_s = stoi(s, est_s, self.srate)
                ideal_stoi_s = stoi(s, ideal_s, self.srate)
                mix_stoi_s = stoi(s, mix, self.srate)
        
                # compute snr
                est_snr_s = snr(s, est_s)
                ideal_snr_s = snr(s, ideal_s)
                mix_snr_s = snr(s, mix)
        
                # compute accu of them
                est_stoi_s_accu += est_stoi_s
                ideal_stoi_s_accu += ideal_stoi_s
                mix_stoi_s_accu += mix_stoi_s
                est_snr_s_accu += est_snr_s
                ideal_snr_s_accu += ideal_snr_s
                mix_snr_s_accu += mix_snr_s
                print('#' * 5)
                print('{}/{}, mix_stoi = {:.4f}, ideal_stoi = {:.4f}, est_stoi = {:.4f}'.format(j +1,
                                                                                                self.num_test_sentences,
                                                                                                mix_stoi_s,
                                                                                                ideal_stoi_s, 
                                                                                                est_stoi_s))
                      
                print('{}/{}, mix_snr = {:.4f}, ideal_snr = {:.4f}, est_snr = {:.4f}'.format(j+1, self.num_test_sentences, 
                                                                                             mix_snr_s, 
                                                                                             ideal_snr_s, 
                                                                                             est_snr_s))
               
                mix_norm, scale_mix = normalize_wav(mix)
                est_s_norm, scale_est_s = normalize_wav(est_s)
                ideal_s_norm, scale_ideal_s = normalize_wav(ideal_s)
                s_norm, scale_s = normalize_wav(s)
                    
                # save them into wav files
                filepath = os.path.join(self.wav_path, self.assess_list[i])
                if not os.path.isdir(filepath):
                    os.makedirs(filepath)
                filename_mix = '%d_mix.wav' % (j)
                filename_s_est = '%d_s_est.wav' % (j)
                filename_s_ideal = '%d_s_ideal.wav' % (j)
                filename_s = '%d_s.wav' % (j)
                    
                sf.write(os.path.join(filepath, filename_mix), mix_norm,int(self.srate))
                sf.write(os.path.join(filepath, filename_s_est), est_s_norm,int(self.srate))
                sf.write(os.path.join(filepath, filename_s_ideal), ideal_s_norm,int(self.srate))
                sf.write(os.path.join(filepath, filename_s), s_norm,int(self.srate))
        
            #bar.finish()
            f_mix.close()
            f_s.close()
            f_est_s.close()
            f_ideal_s.close()
            
            mix_stoi_s_ave = mix_stoi_s_accu / float(self.num_test_sentences)
            ideal_stoi_s_ave = ideal_stoi_s_accu / float(self.num_test_sentences)
            est_stoi_s_ave = est_stoi_s_accu / float(self.num_test_sentences)
            mix_snr_s_ave = mix_snr_s_accu / float(self.num_test_sentences)
            ideal_snr_s_ave = ideal_snr_s_accu / float(self.num_test_sentences)
            est_snr_s_ave = est_snr_s_accu / float(self.num_test_sentences)
            end1 = timeit.default_timer()
            print('')
            print('{}/{}, Finished assess on {}. time taken = {:.4f}'.format(i+1, len(self.assess_list), 
                                                                             self.assess_list[i], end1 - start1))
            print('')
            
            print('#'*100)
            print('#'*100)
            print('mix_stoi: %.4f | ideal_stoi: %.4f | mix_stoi: %.4f' % (mix_stoi_s_ave, ideal_stoi_s_ave, est_stoi_s_ave))
            print('mix_snr: %.2f | ideal_snr: %.2f | mix_snr: %.2f' % (mix_snr_s_ave, ideal_snr_s_ave, est_snr_s_ave))
            print('#'*100)
            print('#'*100)
            
            
    
    
    def getPESQ(self):
        for i in range(len(self.assess_list)):
            #print '\n[%d/%d] Assess on %s...' % ((i+1),len(self.assess_list),self.assess_list[i])  
            start1 = timeit.default_timer()
            print('')
            print('{}/{}, Started assess on {} ...'.format(i+1, len(self.assess_list), self.assess_list[i]))
            print('')
            # create arrays for stoi and snr
            est_pesq_s_accu = 0.0
            ideal_pesq_s_accu = 0.0
            unprocessed_pesq_s_accu = 0.0
            
            for j in range(self.num_test_sentences):
                filepath = os.path.join(self.wav_path, self.assess_list[i])
                filename_mix = '%d_mix.wav' % (j)
                filename_s_est = '%d_s_est.wav' % (j)
                filename_s_ideal = '%d_s_ideal.wav' % (j)
                filename_s = '%d_s.wav' % (j)
    
                unprocessed_pesq_s = getPESQ(self.tool_path, os.path.join(filepath, filename_s),os.path.join(filepath, filename_mix))
                ideal_pesq_s = getPESQ(self.tool_path,os.path.join(filepath, filename_s), os.path.join(filepath, filename_s_ideal))
                est_pesq_s = getPESQ(self.tool_path, os.path.join(filepath, filename_s), os.path.join(filepath, filename_s_est))
        
                # compute accu of them
                est_pesq_s_accu += est_pesq_s
                ideal_pesq_s_accu += ideal_pesq_s
                unprocessed_pesq_s_accu += unprocessed_pesq_s
        
                # update
                #bar.update(j,
                           #mix_pesq=unprocessed_pesq_s_accu/float(j+1),
                           #ideal_pesq=ideal_pesq_s_accu/float(j+1),
                           #est_pesq=est_pesq_s_accu/float(j+1))
                print('#' * 5)
                print('{}/{}, mix_pesq = {:.4f}, ideal_pesq = {:.4f}, '
                      'est_pesq = {:.4f}'.format(j + 1, self.num_test_sentences,
                                                 unprocessed_pesq_s,                                                            
                                                 ideal_pesq_s,
                                                 est_pesq_s))
                    
            #bar.finish()
            end1 = timeit.default_timer()
            print('')
            print('{}/{}, Finished assess on {}. time taken = {:.4f}'.format(i+1, len(self.assess_list), 
                                                                             self.assess_list[i], end1 - start1))
            print('')
            
            print('#'*100)
            print('#'*100)
            mix_pesq_s_ave = unprocessed_pesq_s_accu / float(self.num_test_sentences)
            ideal_pesq_s_ave = ideal_pesq_s_accu / float(self.num_test_sentences)
            est_pesq_s_ave = est_pesq_s_accu / float(self.num_test_sentences)
            print('mix_pesq: %.2f | ideal_pesq: %.2f | mix_pesq: %.2f' % (mix_pesq_s_ave, ideal_pesq_s_ave, est_pesq_s_ave))
            print('#'*100)
            print('#'*100)
