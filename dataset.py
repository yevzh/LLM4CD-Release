from tqdm import tqdm
import os
import pickle
import logging as log
import torch
from torch.utils import data
# from torch_geometric.data import Data, Batch
import math
import random
import json
import numpy as np
from torch.utils.data import DataLoader
class Dataset(data.Dataset):
    def __init__(self, knowledge_dim, split, args):
        self.args = args
        self.split = split
        self.knowledge_dim = knowledge_dim
        self.process()
        
            
    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        return self.data_list[idx]


        

    def process(self):
        if self.split == 'train':
            data_file = self.args.data_dir+'train_set.json'
        elif self.split == 'valid':
            data_file = self.args.data_dir+'val_set.json'
        else:
            data_file = self.args.data_dir+'test_set.json'
        with open(data_file, encoding='utf8') as i_f:
            data_raw = json.load(i_f)
        data = []
        for stu in data_raw:
            records = stu['logs']
            user_id = stu['user_id']
            for log in records:
                data.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
        if self.split == 'train':
            random.shuffle(data) 

        self.data_list = []
        for item in tqdm(range(len(data)), desc="Processing", unit="item"):
            record = data[item]
            new_record = []
            new_record.append(record['user_id']-1)
            new_record.append(record['exer_id']-1)
            new_record.append(record['score'])
            knowledge_emb = [0.]* self.knowledge_dim
            for k in record['knowledge_code']:
                knowledge_emb[k] = 1.0
            new_record.append(torch.tensor(np.array(knowledge_emb)).float())
            
            self.data_list.append(new_record)