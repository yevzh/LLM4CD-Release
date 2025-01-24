import pandas as pd
import os
import jieba
import json
from rank_bm25 import BM25Okapi
from tqdm import trange, tqdm
import numpy as np
import pickle
from utils import *
import random
import faiss
from collections import defaultdict
data_dir = 'text/'

# with open(os.path.join(data_dir, 'ques_text_dict.pkl'), 'rb') as f:
#     ques_text_dic = pickle.load(f)
# with open(os.path.join(data_dir, 'ques_text_sum.pkl'), 'rb') as f:
#     ques_text_sum = pickle.load(f)
with open('id_skill_desc_dict.json', 'rb') as f:
    disc_dic = json.load(f)
with open('id_skillname_dict.json', 'rb') as f:
    name_dic = json.load(f)





data_train = open_json('../../data/ass09/train_set.json')

data_valid = open_json('../../data/ass09/val_set.json')
data_test = open_json('../../data/ass09/test_set.json')

def all_records():
    train_dict = {}
    for data in tqdm(data_train, desc="Processing", unit="data"):
        user_id = data['user_id']
        log_num = data['log_num']
        logs = data['logs']
        template = "A student participated in a **math** test and answered the following concepts\n"
        cnt = 1
        for stu_log in logs:
            knowledge_list = stu_log['knowledge_code']
            if stu_log["score"] == 1:
                resp = "correct"
            else:
                resp = "wrong"
            # print(len(knowledge_list))

            for kn in knowledge_list:
                # if kn == 0:
                #     print(1)
                knowledge_name = name_dic[str(kn)]

                template = template + "Concept: {}, Response:{};\n".format(knowledge_name, resp)

                cnt+=1


        train_dict[user_id] = template  
        
        
    
    dump_json('student_record.json', train_dict)



def retrieve_records():
    train_dict = {}
    for data in tqdm(data_train, desc="Processing", unit="data"):
        user_id = data['user_id']
        log_num = data['log_num']
        logs = data['logs']
        stu_text_list = []
        stu_text_info = []
        stu_log_text = {}
        gt = []
        for i in range(log_num):
            stu_text_list.append(name_dic[str(logs[i]['knowledge_code'][0])])
            stu_text_info.append(disc_dic[str(logs[i]['knowledge_code'][0])])
            gt.append(logs[i]['score'])
        for i in range(log_num):
            log = logs[i]
            tmp = stu_text_list[i]
            stu_text_list[i] = 'mask'
            similar_list = get_prompt_bm25(stu_text_list, name_dic[str(log['knowledge_code'][0])], 10)
            txt = generate_prompt(similar_list, stu_text_info, gt)
            stu_log_text[log['exer_id']] = txt
            stu_text_list[i] = tmp
        train_dict[user_id] = stu_log_text  
    dump_json('train_ass09.json', train_dict)
    valid_dict = {}
    for item in tqdm(range(len(data_train)), desc="Processing", unit="data"):
        data = data_train[item]
        log_num = data['log_num']
        logs = data['logs']
        stu_text_list = []
        stu_text_info = []
        stu_log_text = {}
        gt = []
        for i in range(log_num):
            stu_text_list.append(name_dic[str(logs[i]['knowledge_code'][0])])
            stu_text_info.append(disc_dic[str(logs[i]['knowledge_code'][0])])
            gt.append(logs[i]['score'])
        data = data_valid[item]
        user_id = data['user_id']
        log_num = data['log_num']
        logs = data['logs']
        for i in range(log_num):
            log = logs[i]

            similar_list = get_prompt_bm25(stu_text_list, name_dic[str(log['knowledge_code'][0])], 10)
            txt = generate_prompt(similar_list, stu_text_info, gt)
            stu_log_text[log['exer_id']] = txt
        valid_dict[user_id] = stu_log_text  
    dump_json('valid_ass09.json', valid_dict)    
    test_dict = {}
    for item in tqdm(range(len(data_train)), desc="Processing", unit="data"):
        data = data_train[item]
        log_num = data['log_num']
        logs = data['logs']
        stu_text_list = []
        stu_text_info = []
        stu_log_text = {}
        gt = []
        for i in range(log_num):
            stu_text_list.append(name_dic[str(logs[i]['knowledge_code'][0])])
            stu_text_info.append(disc_dic[str(logs[i]['knowledge_code'][0])])
            gt.append(logs[i]['score'])
        data = data_test[item]
        user_id = data['user_id']
        log_num = data['log_num']
        logs = data['logs']
        for i in range(log_num):
            log = logs[i]

            similar_list = get_prompt_bm25(stu_text_list, name_dic[str(log['knowledge_code'][0])], 10)
            txt = generate_prompt(similar_list, stu_text_info, gt)
            stu_log_text[log['exer_id']] = txt
        test_dict[user_id] = stu_log_text  
    dump_json('test_ass09.json', test_dict)    
    
    
def generate_prompt(similarity_list, stu_text_list, gt_list):
    # my_list = [1, 2, 2, 3, 4, 4, 5]
    sim_list = []
    for item in similarity_list:
        if item not in sim_list:
            sim_list.append(item)
    template = "The student answers concepts:\n"
    sim_len = len(sim_list)
    # print(sim_list)
    for i in range(sim_len):
        if gt_list[sim_list[i]] == 0:
            resp = "correct"
        else:
            resp = "wrong"
        template = template + "|{}.description: {}; response: {};\n".format(i, stu_text_list[sim_list[i]], resp)
    # print(template)
    return template       
    
def get_prompt_bm25(documents, query, top_k):
    
    corpus = [list(jieba.cut(doc)) for doc in documents]
    query = list(jieba.cut(query)) 
    model = BM25Okapi(corpus)
    scores = model.get_scores(query)
    # print(model.get_top_n(query, corpus, n=top_k))
    top_index = [corpus.index(qcut) for qcut in model.get_top_n(query, corpus, n=top_k)]
    # prompt_list = [{'question': documents[corpus.index(qcut)]} for qcut in model.get_top_n(query, corpus, n=top_k) if scores[corpus.index(qcut)]>10]
    return top_index

all_records()
# generate_student_prompt()
# retrieve_records()
# two_records()