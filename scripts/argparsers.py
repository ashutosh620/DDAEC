# -*- coding: utf-8 -*-

import gflags
import os
class ArgParser(object):
    
    def __init__(self, argv, mode):
        if mode == 'train':
            FLAGS = gflags.FLAGS
            gflags.DEFINE_string('train_list', '', 'training list')
            gflags.DEFINE_string('evaluate_file', '', 'evaluate file')
            gflags.DEFINE_integer('display_eval_steps', '250', 'display_eval_steps')
            gflags.DEFINE_integer('eval_plot_num','3','eval_plot_num')
            gflags.DEFINE_string('resume_model', '', 'resume_model')
            gflags.DEFINE_string('model_name', '', 'model_name')
            gflags.DEFINE_integer('width','64','width of network')
            gflags.DEFINE_integer('batch_size','16','batch_size')
            FLAGS(argv)
        
            self.train_list = FLAGS.train_list
            self.eval_file = FLAGS.evaluate_file
            self.eval_steps = FLAGS.display_eval_steps
            self.eval_plot_num = FLAGS.eval_plot_num
            self.resume_model = FLAGS.resume_model
            self.model_name = FLAGS.model_name
            self.width = FLAGS.width
            self.batch_size = FLAGS.batch_size
        elif mode == 'test':
            FLAGS = gflags.FLAGS
            gflags.DEFINE_string('test_list', '', 'test list')
            gflags.DEFINE_string('model_file', '', 'model file')
            gflags.DEFINE_string('model_name', '', 'model_name')
            gflags.DEFINE_integer('width','64','width')
            FLAGS(argv)
        
            self.test_list = FLAGS.test_list
            self.model_file = FLAGS.model_file
            self.model_name = FLAGS.model_name
            self.width = FLAGS.width 
        elif mode == 'assess':
            FLAGS = gflags.FLAGS
            gflags.DEFINE_string('assess_list', '', 'assess list')
            gflags.DEFINE_string('model_name', '', 'model_name')
            FLAGS(argv)
            
            self.assess_list = FLAGS.assess_list
            self.model_name = FLAGS.model_name
        else:
            raise ValueError('Invalid mode.')
            


class Args(object):
    r"""Arguments."""
    
    def __init__(self, parser, mode):
        root_path = '../../DDAEC/' # the directory to stores train and test mixtures and model files 
        if mode == 'train':
            # args from parser
            
            self.train_list = parser.train_list
            self.eval_file = parser.eval_file
            self.eval_steps = parser.eval_steps
            self.eval_plot_num = parser.eval_plot_num
            self.resume_model = parser.resume_model
            self.model_name = parser.model_name
            self.width = parser.width
            # file paths
            self.log_path = '../logs/'
            if not os.path.isdir(self.log_path):
                os.makedirs(self.log_path)
            self.stat_path = os.path.join(root_path, 'data', 'stat')
            self.model_path = os.path.join(root_path, 'models', self.model_name)
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)
            self.fig_path = os.path.join(root_path, 'figures', self.model_name)
            self.wav_path = os.path.join('../waves/', self.model_name, 'val')
            if not os.path.isdir(self.wav_path):
                os.makedirs(self.wav_path)
            # hyperparameters
            self.num_train_sentences = 320000
            self.num_test_sentences = 150
            self.mean_var_num = 1000 # calculate mean and variance from 1000 training examples
            self.lr = 0.0002
            self.batch_size = parser.batch_size
            self.frame_size = 512
            self.srate = 16000.0
            self.max_epoch = 15
            self.tool_path = './bin/'
        elif mode == 'test':
            # args from parser
            self.test_list = parser.test_list
            self.model_file = parser.model_file
            self.model_name = parser.model_name
            self.width = parser.width
            self.frame_size = 512
            # file paths
            self.stat_path = os.path.join(root_path, 'data', 'stat') 
            self.test_mixture_path = '../data/mixture/test/'
            self.prediction_path = os.path.join(root_path, 'data', 'prediction', self.model_name)
            if not os.path.isdir(self.prediction_path):
                os.makedirs(self.prediction_path)
            # hyperparameters
            self.num_test_sentences = 150
            self.srate = 16000.0
        elif mode == 'assess':
            # args from parser
            self.assess_list = parser.assess_list
            self.model_name = parser.model_name
            # file paths
            self.tool_path = './bin/'
            self.wav_path = os.path.join('../waves/', self.model_name)
            if not os.path.isdir(self.wav_path):
                os.makedirs(self.wav_path)
            self.test_mixture_path = '../data/mixture/test/'
            self.prediction_path = os.path.join(root_path, 'data', 'prediction', self.model_name)
            # hyperparameters
            self.num_test_sentences = 150
            self.srate = 16000.0
        else:
            raise ValueError('Invalid mode.')
