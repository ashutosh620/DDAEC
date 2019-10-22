# -*- coding: utf-8 -*-
import sys

import torch

from models import Model
from argparsers import ArgParser, Args

def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parse commands
    parser = ArgParser(sys.argv, mode='test')
    args = Args(parser, mode='test')
    
    
    # train the model
    model = Model(args)
    model.test(args)
    

if __name__ == '__main__':
    main()
