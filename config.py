#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Parse input command to hyper-parameters

import ast
import argparse

parser = argparse.ArgumentParser()
arg_list = []

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')

data_arg.add_argument('--min_n', type=int, default=3, help='')
data_arg.add_argument('--max_n', type=int, default=4, help='')
data_arg.add_argument('--n', type=int, default=3, help='')
data_arg.add_argument('--hidden_d', type=int, default=1024, help='')
data_arg.add_argument('--ablation_flag', type=int, default=0, help='0:no ablation; 1:ablation no attention; 2:ablation no weight in loss')
data_arg.add_argument('--times_n', type=int, default=1, help='run the code how many times')
data_arg.add_argument('--O', type=int, default=0, help='0:no scal; >0:scal branch for varying O')
data_arg.add_argument('--T', type=int, default=0, help='0:no scal; >0:scal branch for varying T')

data_arg.add_argument('--raw_n', type=str, default='11_01_11_07', help='')
data_arg.add_argument('--file_name', type=str, default='xx', help='')
data_arg.add_argument('--interval', type=int, default=2, help='')

data_arg.add_argument('--group_size', type=int, default=4, help='')
data_arg.add_argument('--train_ratio', type=float, default=0.7, help='')
data_arg.add_argument('--val_ratio', type=float, default=0.1, help='')

data_arg.add_argument('--num_heads', type=int, default=2, help='')
data_arg.add_argument('--max_epochs', type=int, default=100, help='')
# to gpu device
data_arg.add_argument('--device', type=int, default=0, help='')

data_arg.add_argument('--starting_ts', type=str, default='2020-11-01 00:00:00', help='')
data_arg.add_argument('--ending_ts', type=str, default='2020-11-07 23:59:59', help='')
data_arg.add_argument('--limit_d', type=int, default=5000000, help='')

data_arg.add_argument('--neg_ratio', type=int, default=3, help='')
data_arg.add_argument('--batch_size', type=int, default=128, help='')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

