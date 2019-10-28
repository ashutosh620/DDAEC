# -*- coding: utf-8 -*-

import os
import shutil

import numpy as np
import h5py
import torch
torch.backends.cudnn.benchmark = True
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progressbar import ProgressBar, Bar, Timer, ETA, Percentage, DynamicMessage

from datasets import TrainingDataset, EvalDataset, ToTensor, TrainCollate, EvalCollate, SignalToFrames, OLA, ToTensor
from networks import Net
from criteria import mse_loss, stftm_loss
from utils import compLossMask, numParams, logging, plotting, metric_logging
import timeit
import sys
import librosa
from STOI import stoi
from PESQ import getPESQ
from metrics import snr, normalize_wav
import soundfile as sf
class Checkpoint(object):
    def __init__(self, start_epoch=None, start_iter=None, train_loss=None, eval_loss=None, best_loss=np.inf, state_dict=None, optimizer=None):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.best_loss = best_loss
        self.state_dict = state_dict
        self.optimizer = optimizer
    
    
    def save(self, is_best, filename, best_model):
        print('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)
        if is_best:
            print('Saving the best model at "%s"' % best_model)
            shutil.copyfile(filename, best_model)
        print('\n')


    def load(self, filename):
        if os.path.isfile(filename):
            print('Loading checkpoint from "%s"\n' % filename)
            checkpoint = torch.load(filename, map_location='cpu')
            
            self.start_epoch = checkpoint.start_epoch
            self.start_iter = checkpoint.start_iter
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss
            self.best_loss = checkpoint.best_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)



class Model(object):
    def __init__(self, args):
        #self.feature_mean = stat_data['feature_mean']
        #self.feature_variance = stat_data['feature_variance']
        self.num_test_sentences = args.num_test_sentences
        self.model_name = args.model_name
        self.frame_size = args.frame_size
        self.frame_shift = args.frame_size // 2
        self.get_frames = SignalToFrames(frame_size=self.frame_size, frame_shift=self.frame_shift)
        self.ola = OLA(frame_shift=self.frame_shift)
        self.to_tensor = ToTensor()
        self.width = args.width
        self.srate = 16000
            
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        

        
    def train(self, args):
        with open(args.train_list,'r') as train_list_file:
            self.train_list = [line.strip() for line in train_list_file.readlines()]
        self.eval_file = args.eval_file
        self.num_train_sentences = args.num_train_sentences
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.max_epoch = args.max_epoch
        self.model_path = args.model_path
        self.log_path = args.log_path
        self.fig_path = args.fig_path
        self.eval_plot_num = args.eval_plot_num
        self.eval_steps = args.eval_steps
        self.resume_model = args.resume_model
        self.wav_path = args.wav_path
        self.tool_path = args.tool_path
        
        # create a training dataset and an evaluation dataset
        trainSet = TrainingDataset(self.train_list,
                                   frame_size=self.frame_size,
                                   frame_shift=self.frame_shift)
        evalSet = EvalDataset(self.eval_file,
                              self.num_test_sentences)
        #trainSet = evalSet   
        # create data loaders for training and evaluation
        train_loader = DataLoader(trainSet,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=16,
                                  collate_fn=TrainCollate())
        eval_loader = DataLoader(evalSet,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=EvalCollate())
        
        # create a network
        print('model', self.model_name)
        net = Net(device=self.device, L=self.frame_size, width=self.width)
        #net = torch.nn.DataParallel(net)
        net.to(self.device)
        print('Number of learnable parameters: %d' % numParams(net))
        print(net)
        
        criterion = mse_loss()
        criterion1 = stftm_loss(device=self.device)
        optimizer = torch.optim.Adam(net.parameters(),lr=self.lr)
        self.lr_list = [0.0002]*3 + [0.0001]*6 + [0.00005]*3 + [0.00001]*3
        if self.resume_model:
            print('Resume model from "%s"' % self.resume_model)
            checkpoint = Checkpoint()
            checkpoint.load(self.resume_model)
            start_epoch = checkpoint.start_epoch
            start_iter = checkpoint.start_iter
            best_loss = checkpoint.best_loss
            net.load_state_dict(checkpoint.state_dict)
            optimizer.load_state_dict(checkpoint.optimizer)
        else:
            print('Training from scratch.')
            start_epoch = 0
            start_iter = 0
            best_loss = np.inf
        
        
        
        num_train_batches = self.num_train_sentences // self.batch_size
        total_train_batch = self.max_epoch * num_train_batches
        print('num_train_sentences', self.num_train_sentences)
        print('batches_per_epoch', num_train_batches)
        print('total_train_batch', total_train_batch)
        print('batch_size', self.batch_size)
        print('model_name', self.model_name)
        batch_timings = 0.
        counter = int(start_epoch*num_train_batches + start_iter)
        counter1 = 0
        print('counter', counter)
        ttime = 0.
        cnt = 0.
        print('best_loss', best_loss)
        for epoch in range(start_epoch, self.max_epoch):
            accu_train_loss = 0.0
            net.train()
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr_list[epoch]
                
            start  = timeit.default_timer()
            for i, (features, labels, nframes) in enumerate(train_loader):
                i += start_iter
                features, labels = features.to(self.device), labels.to(self.device)
                
                loss_mask = compLossMask(labels, nframes=nframes)
                
                # forward + backward + optimize
                optimizer.zero_grad()
                
                outputs = net(features)
                outputs = outputs[:, :, :labels.shape[-1]]
                
                loss1 = criterion(outputs, labels, loss_mask, nframes)
                loss2 = criterion1(outputs, labels, loss_mask, nframes)
                
                loss = 0.8*loss1 + 0.2*loss2
                loss.backward()
                optimizer.step()
                # calculate losses
                running_loss = loss.data.item()
                accu_train_loss += running_loss
                
                cnt += 1.
                counter += 1
                counter1 += 1
                
                del loss, loss1, loss2, outputs, loss_mask, features, labels
                end = timeit.default_timer()
                curr_time = end - start
                ttime += curr_time
                mtime = ttime / counter1
                print('{}/{} (epoch = {}), loss = {:.5f}, time/batch = {:.5f}, mtime/batch = {:.5f}'.format(counter, 
                total_train_batch, epoch, running_loss, curr_time, mtime))
                start  = timeit.default_timer()
                if (i+1) % self.eval_steps == 0:
                    start = timeit.default_timer()
                    
                    avg_train_loss = accu_train_loss / cnt
                    
                    avg_eval_loss = self.validate(net, eval_loader)
                    
                    net.train()
                    
                    print('Epoch [%d/%d], Iter [%d/%d]  ( TrainLoss: %.4f | EvalLoss: %.4f )' % (epoch+1,self.max_epoch,i+1,self.num_train_sentences//self.batch_size,avg_train_loss,avg_eval_loss))
                    
                    is_best = True if avg_eval_loss < best_loss else False
                    best_loss = avg_eval_loss if is_best else best_loss
                    
                    checkpoint = Checkpoint(epoch, i, avg_train_loss, avg_eval_loss, best_loss, net.state_dict(), optimizer.state_dict())
                    
                    model_name = self.model_name + '_latest.model'
                    best_model = self.model_name + '_best.model'
                    checkpoint.save(is_best, os.path.join(self.model_path, model_name), os.path.join(self.model_path, best_model))
                    
                    logging(self.log_path, self.model_name +'_loss_log.txt', checkpoint, self.eval_steps)
                    #metric_logging(self.log_path, self.model_name +'_metric_log.txt', epoch+1, [avg_st, avg_sn, avg_pe]) 
                    accu_train_loss = 0.0
                    cnt = 0.
                    
                    net.train()
                if (i+1)%num_train_batches == 0:
                    break
                
            avg_st, avg_sn, avg_pe = self.validate_with_metrics(net, eval_loader)
            net.train()
            print('#'*50)
            print('')
            print('After {} epoch the performance on validation score is a s follows:'.format(epoch+1))
            print('')
            print('STOI: {:.4f}'.format(avg_st))
            print('SNR: {:.4f}'.format(avg_sn))
            print('PESQ: {:.4f}'.format(avg_pe))
            for param_group in optimizer.param_groups:
                print('learning_rate', param_group['lr'])
            print('')
            print('#'*50)
            checkpoint = Checkpoint(epoch, 0, None, None, best_loss, net.state_dict(), optimizer.state_dict())
            checkpoint.save(False, os.path.join(self.model_path, self.model_name + '-{}.model'.format(epoch+1)),
                            os.path.join(self.model_path, best_model))
            metric_logging(self.log_path, self.model_name +'_metric_log.txt', epoch, [avg_st, avg_sn, avg_pe])
            start_iter = 0.
                
                    
                    
    def validate(self, net, eval_loader):
        #print('********** Started evaluation on validation set ********')
        net.eval()
        
        with torch.no_grad():
            mtime = 0
            ttime = 0.
            cnt = 0.
            accu_eval_loss = 0.0
            for k, (mix_raw, cln_raw) in enumerate(eval_loader):
                start = timeit.default_timer()
            
                est_s = self.eval_forward(mix_raw, net)
                est_s = est_s[:mix_raw.size]
                #print('mix_raw', mix_raw.shape, 'cln_raw', cln_raw.shape, 'est_s', est_s.shape)
                
                eval_loss = np.mean((est_s-cln_raw)**2)
                accu_eval_loss += eval_loss
                
                cnt += 1.
                
                end = timeit.default_timer()
                curr_time = end - start
                ttime += curr_time
                mtime = ttime / cnt
                
                #print('{}/{}, eval_loss = {:.4f}, time/utterance = {:.4f}, '
                #      'mtime/utternace = {:.4f}'.format(k+1, self.num_test_sentences, eval_loss, curr_time, mtime))
                
            avg_eval_loss = accu_eval_loss / cnt
        net.train()
        return avg_eval_loss
                
            
    def validate_with_metrics(self, net, eval_loader):
        print('********** Started metrics evaluation on validation set ********')
        accu_stoi = 0.0
        accu_pesq = 0.0
        accu_snr = 0.0
        net.eval()
        
        with torch.no_grad():
            mtime = 0.
            ttime = 0.
            count = 0.
            for k, (mix_raw, cln_raw) in enumerate(eval_loader):
    
                start = timeit.default_timer()
                est_s = self.eval_forward(mix_raw, net)
                est_s = est_s[:mix_raw.size]
                ideal_s = cln_raw
                st =  stoi(cln_raw, est_s, self.srate)
                sn = snr(cln_raw, est_s)
                accu_stoi += st
                accu_snr += sn
                est_path = os.path.join(self.wav_path, '{}_est.wav'.format(k+1))
                ideal_path = os.path.join(self.wav_path, '{}_ideal.wav'.format(k+1))
                sf.write(est_path, normalize_wav(est_s)[0], self.srate)
                sf.write(ideal_path, normalize_wav(ideal_s)[0], self.srate)
                pe = getPESQ(self.tool_path, ideal_path, est_path)
                accu_pesq += pe
                count += 1
                #print('{}, STOI: {:.4f}, SNR: {:.4f}, PESQ: {:.4f}'.format(k+1, st, sn, pe))
            avg_stoi = accu_stoi / count
            avg_snr = accu_snr / count
            avg_pesq = accu_pesq / count
        net.train()
        return avg_stoi, avg_snr, avg_pesq
            
            
    def eval_forward(self, mix_raw, net):
        feature = self.to_tensor(self.get_frames(np.reshape(mix_raw, [1, 1, -1])))
        feature = feature.to(self.device)
        output = net(feature)
        output1 = output.cpu().numpy()[0][0]
        del output, feature
        return output1
        
        
    def test(self, args):
        with open(args.test_list,'r') as test_list_file:
            self.test_list = [line.strip() for line in test_list_file.readlines()]
        self.model_name = args.model_name
        self.model_file = args.model_file
        self.test_mixture_path = args.test_mixture_path
        self.prediction_path = args.prediction_path
        
        # create a network
        print('model', self.model_name)
        net = Net(device=self.device, L=self.frame_size, width=self.width)
        #net = torch.nn.DataParallel(net)
        net.to(self.device)
        print('Number of learnable parameters: %d' % numParams(net))
        print(net)
        # loss and optimizer
        criterion = mse_loss()
        net.eval()
        print('Load model from "%s"' % self.model_file)
        checkpoint = Checkpoint()
        checkpoint.load(self.model_file)
        net.load_state_dict(checkpoint.state_dict)
        with torch.no_grad():
            for i in range(len(self.test_list)):
                # read the mixture for resynthesis
                filename_input = self.test_list[i].split('/')[-1]
                start1 = timeit.default_timer()
                print('{}/{}, Started working on {}.'.format(i+1, len(self.test_list), self.test_list[i]))
                print('')
                filename_mix = filename_input.replace('.samp', '_mix.dat')

                filename_s_ideal = filename_input.replace('.samp', '_s_ideal.dat')
                filename_s_est = filename_input.replace('.samp', '_s_est.dat')
                #print(filename_mix)
                #sys.exit()
                f_mix = h5py.File(os.path.join(self.test_mixture_path, filename_mix),'r')
                f_s_ideal = h5py.File(os.path.join(self.prediction_path, filename_s_ideal),'w')
                f_s_est = h5py.File(os.path.join(self.prediction_path, filename_s_est),'w')
                # create a test dataset
                testSet = EvalDataset(os.path.join(self.test_mixture_path, self.test_list[i]),
                                      self.num_test_sentences)

                # create a data loader for test
                test_loader = DataLoader(testSet,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=2,
                                         collate_fn=EvalCollate())


                #print '\n[%d/%d] Predict on %s' % (i+1, len(self.test_list), self.test_list[i])

                accu_test_loss = 0.0
                accu_test_nframes = 0

                ttime = 0.
                mtime = 0.
                cnt = 0.
                for k, (mix_raw, cln_raw) in enumerate(test_loader):
                    start = timeit.default_timer()
                    est_s = self.eval_forward(mix_raw, net)
                    est_s = est_s[:mix_raw.size]
                    mix = f_mix[str(k)][:]
                    
                    ideal_s = cln_raw
                    
                    f_s_ideal.create_dataset(str(k), data=ideal_s.astype(np.float32),chunks=True)
                    f_s_est.create_dataset(str(k), data=est_s.astype(np.float32),chunks=True)
                    # compute eval_loss                        

                    test_loss = np.mean((est_s-ideal_s)**2)

                    accu_test_loss += test_loss
                    cnt += 1
                    end = timeit.default_timer()
                    curr_time = end - start
                    ttime += curr_time
                    mtime = ttime / cnt
                    mtime = (mtime * (k) + (end-start)) / (k+1)
                    print('{}/{}, test_loss = {:.4f}, time/utterance = {:.4f}, '
                           'mtime/utternace = {:.4f}'.format(k+1, self.num_test_sentences, test_loss, curr_time, mtime))
                    
                avg_test_loss = accu_test_loss / cnt
                    #bar.update(k,test_loss=avg_test_loss)
                #bar.finish()
                end1 = timeit.default_timer()
                print('********** Finisehe working on {}. time taken = {:.4f} **********'.format(filename_input,
                                                                                                 end1 - start1))
                print('')
                f_mix.close()
                f_s_est.close()
                f_s_ideal.close()
