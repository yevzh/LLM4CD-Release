#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader
import multiprocessing
import os
import pickle
import argparse
import logging as log

# import models
import model
import importlib

from dataset import Dataset
import numpy as np
import random
import math
import wandb


from train_lamda import train_lamda

torch.set_num_threads(60)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


##################################################################

parser = argparse.ArgumentParser(description='Hidden Graph Model')
parser.add_argument('--dataset', type=str, default='ass09')
parser.add_argument('--device',         type=int,   default=1,      help='gpu device id, cpu if -1')
parser.add_argument('--num_epochs',        type = int, default = 5, help = 'the number of training epochs')
parser.add_argument('--lr', type=float , default=1e-4, help = 'learning rate')
parser.add_argument('--weight_decay', type=float, default= 1e-3)
parser.add_argument('--model',          type=str,   default='ncdm',   help='run model')
parser.add_argument('--llm', type=str, default='chatglm2')
parser.add_argument('--batch_size',     type=int,   default=32,      help='number of instances in a batch')
parser.add_argument('--seed',       type=int, default=20,   help='seed')
parser.add_argument('--save_models', action='store_true')
parser.add_argument('--saved_model', type= str, default='best_model_dict', help='The saved model dictionary for one experiment')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--graph_pretrain', action='store_true')
parser.add_argument('--sgl_temperature', type=float, default = 0.1)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--rate', type=float, default=0.01)
parser.add_argument('--load_embeddings_name', type=str, default='best_model_dict')
parser.add_argument('--stu_text', type=str, default='id')
parser.add_argument('--add_graph', type=str, default='1')
parser.add_argument('--cl_method', type=str, default= 'graph')
parser.add_argument('--retrieval', type = str, default= 'swing')
parser.add_argument('--mode', type=str, default='first')
parser.add_argument('--use_attention', action = 'store_true')
parser.add_argument('--graph_layer', type=int, default=1)
parser.add_argument('--encoder', type=str, default='graph')
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--add_text_emb', action='store_true')
parser.add_argument('--mf_type', type=str, default='gmf')
parser.add_argument('--value_range', type = float, default=None)
parser.add_argument('--fusion_method', type = str, default='add')
parser.add_argument('--ques_emb', type=str, default='desc')
parser.add_argument('--stu_emb', type=str, default='origin')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--ablation', type=str, default='origin')
parser.add_argument('--test', action='store_true')
args = parser.parse_args() 


setup_seed(args.seed)

args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)


if __name__ == '__main__':
    args.data_dir =  f'data/{args.dataset}/'
    args.text_dir =  f'text/{args.dataset}/'
    if 'ass09' in args.data_dir:
        n_stu = 1088
        args.n_stu = n_stu
        n_exer = 15279
        args.n_exer = n_exer
        n_knowledge = 136
        args.n_knowledge = n_knowledge
    elif 'ass12' in args.data_dir:
        n_stu = 2550
        args.n_stu = n_stu
        n_exer = 45706
        args.n_exer = n_exer
        n_knowledge = 245
        args.n_knowledge = n_knowledge
    elif 'boyu' in args.data_dir:
        n_stu = 2337
        args.n_stu = n_stu
        n_exer = 1256
        args.n_exer = n_exer
        n_knowledge = 116
        args.n_knowledge = n_knowledge
    elif 'junyi' in args.data_dir:
        n_stu = 3118
        args.n_stu = n_stu
        n_exer = 2791
        args.n_exer = n_exer
        n_knowledge = 723
        args.n_knowledge = n_knowledge
    if args.wandb:
        wandb.init(
            project="llm4cd",
            name=args.saved_model,
            config =args
        )
    TrainDataset = Dataset(n_knowledge, 'train', args)
    TrainLoader = DataLoader(TrainDataset, batch_size=args.batch_size, shuffle=True)
    ValidDataset = Dataset(n_knowledge, 'valid', args)
    ValidLoader = DataLoader(ValidDataset, batch_size = args.batch_size, shuffle=False)
    TestDataset = Dataset(n_knowledge, 'test', args)
    TestLoader = DataLoader(TestDataset, batch_size = args.batch_size, shuffle=False)
    train_lamda(args, TrainLoader,ValidLoader,TestLoader, n_stu, n_exer, n_knowledge)
                
                
                
            
            
    
