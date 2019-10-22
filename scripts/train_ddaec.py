# -*- coding: utf-8 -*-
import sys

import torch

from models import Model
from argparsers import ArgParser, Args


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parse commands
    parser = ArgParser(sys.argv, mode='train')
    args = Args(parser, mode='train')
    
    # train the model
    model = Model(args)
    model.train(args)
    

if __name__ == '__main__':
    main()
