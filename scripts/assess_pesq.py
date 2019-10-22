# -*- coding: utf-8 -*-
import sys

from argparsers import ArgParser, Args
from metrics import Metric


def main():
    # parse commands
    parser = ArgParser(sys.argv, mode='assess')
    args = Args(parser, mode='assess')
    
    # calculate STOI and SNR scores
    metric = Metric(args)
    metric.getPESQ()
    

if __name__ == '__main__':
    main()
