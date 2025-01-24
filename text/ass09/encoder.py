import os
import torch
import numpy as np
import pickle as pkl
import argparse
from fastchat.model import load_model, get_conversation_template, add_model_args
from sentence_transformers import SentenceTransformer
# from config import *
from typing import Dict, Tuple, Union, Optional

import torch
from torch.nn import Module

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import pipeline
from tqdm import tqdm
from config import *
from utils import *
import pickle
text_dir = 'text/ass09/'
data_dir = 'text/ass09/'



parser =  argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--model_name", type=str, default='cn-alpaca-7B')
parser.add_argument("--temperature", type=float, default=0.01)
parser.add_argument('--debug', action='store_true')
parser.add_argument("--device_id", type=int, default=-1)
# parser.add_argument('--model_name',type=str,default='13b')

args = parser.parse_args()
args.device = 'cpu' if args.device_id < 0 else 'cuda:%i' % args.device_id
# print(args.device, 'gggggggggggggggggg')
args.device = torch.device(args.device)




cur_embed = None
def hook(module, input, output):
    global cur_embed, embeds
    input = input[0].cpu().detach().numpy()
    cur_embed = input
    
def get_embed(model, tokenizer, x):
    if args.model_name == 'bert':
        return model.encode(x)
    x = tokenizer(x, return_tensors="pt", return_attention_mask=True).to(args.device)
    outputs = model(**x, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states 
    with torch.no_grad():
        embedding = list(outputs.hidden_states)
        last_hidden_states = embedding[-1].cpu().numpy()
        last_hidden_states = np.squeeze(last_hidden_states)
        last_hidden_avg_states = np.mean(last_hidden_states, axis=0)
    return last_hidden_avg_states
    
    # return text_encoding.cpu()
def get_text_emb():
    os.makedirs(os.path.join(text_dir, args.model_name), exist_ok=True)
    model, tokenizer = Load_Model_and_Tokenizer(args.model_name, args.device)
    model.to(torch.device(args.device))
    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')

    with open(os.path.join(data_dir, 'ques_text_dict.pkl'), 'rb') as f:
        ques_text_dic = pickle.load(f)
    text_vec_dict = {}
    print(len(ques_text_dic))
    for key, value in tqdm(ques_text_dic.items(), desc="Processing"):
        text = value[1]
        vec = get_embed(model, tokenizer,text)
        text_vec_dict[key] = vec
    with open(os.path.join(os.path.join(text_dir, args.model_name), 'ques_text_emb.pkl'),'wb') as f:
        pickle.dump(text_vec_dict, f) 
    return text_vec_dict

def get_sum_emb():
    os.makedirs(os.path.join(text_dir, args.model_name), exist_ok=True)
    model, tokenizer = Load_Model_and_Tokenizer(args.model_name, args.device)
    model.to(torch.device(args.device))
    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')

    with open(os.path.join(data_dir, 'ques_text_sum.pkl'), 'rb') as f:
        ques_text_dic = pickle.load(f)
    text_vec_dict = {}
    print(len(ques_text_dic))
    for key, value in tqdm(ques_text_dic.items(), desc="Processing"):
        text = value[1]
        vec = get_embed(model, tokenizer,text)
        text_vec_dict[key] = vec
    with open(os.path.join(os.path.join(text_dir, args.model_name), 'ques_sum_emb.pkl'),'wb') as f:
        pickle.dump(text_vec_dict, f) 
    return text_vec_dict

def get_full_text_emb():
    os.makedirs(os.path.join(text_dir, args.model_name), exist_ok=True)
    model, tokenizer = Load_Model_and_Tokenizer(args.model_name, args.device)
    model.to(torch.device(args.device))
    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')

    with open(os.path.join(data_dir, 'ques_text.json'), 'rb') as f:
        text = json.load(f)
        
    stu_all_info = {}
    for key, value in tqdm(text.items(), desc="Processing"):
        stu_list = []
        length = text[key][0]
        stu_records_list = []
        stu_records = text[key][1]
        for i in range(len(stu_records)):
            vec = get_embed(model, tokenizer,stu_records[i])
            stu_records_list.append(vec)
        stu_list.append(length)
        stu_list.append(stu_records_list)
        stu_all_info[key] = stu_list
    with open('embedding_10_text.pkl','wb') as f:
        pickle.dump(stu_all_info, f) 
    # with open(os.path.join(data_dir, 'test.json'), 'rb') as f:
    #     text = json.load(f)
        
    # stu_all_info = {}
    # for key, value in tqdm(text.items(), desc="Processing"):
    #     stu_list = []
    #     length = text[key][0]
    #     stu_records_list = []
    #     stu_records = text[key][1]
    #     for i in range(len(stu_records)):
    #         vec = get_embed(model, tokenizer,stu_records[i])
    #         stu_records_list.append(vec)
    #     stu_list.append(length)
    #     stu_list.append(stu_records_list)
    #     stu_all_info[key] = stu_list
    # with open('embedding_20.pkl','wb') as f:
    #     pickle.dump(stu_all_info, f) 
    
def get_full_text_emb_dict():
    os.makedirs(os.path.join(text_dir, args.model_name), exist_ok=True)
    model, tokenizer = Load_Model_and_Tokenizer(args.model_name, args.device)
    model.to(torch.device(args.device))
    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')
    splits = [ 'valid', 'test', 'train' ]
    for split in splits:
        with open(data_dir+split+'_ass09.json','rb') as f:
            text = json.load(f)
        
            
        stu_all_info = {}
        print(len(text))
        for key, value in tqdm(text.items(), desc="Processing"):
            stu_emb_info = {}
            stu_dict = text[key]
            for k, v in stu_dict.items():
                vec = get_embed(model, tokenizer,v)
                stu_emb_info[k] = vec
            stu_all_info[key] = stu_emb_info
                
        with open(os.path.join(text_dir, args.model_name)+ '/'+ split + '_embedding_ass09.pkl','wb') as f:
            pickle.dump(stu_all_info, f) 
            
def get_ass_stu_emb():
    os.makedirs(os.path.join(text_dir, args.model_name), exist_ok=True)
    
    model, tokenizer = Load_Model_and_Tokenizer(args.model_name, args.device)
    model.to(torch.device(args.device))
    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')

    with open('text/ass09/student_record.json','rb') as f:
        text = json.load(f)
    
            
    stu_all_info = {}
    for key, value in tqdm(text.items(), desc="Processing"):

        stu_emb_info = get_embed(model, tokenizer,value)
        stu_all_info[key] = stu_emb_info
            
    with open(os.path.join(text_dir, args.model_name)+'/student_embedding.pkl','wb') as f:
        pickle.dump(stu_all_info, f) 
        
            
def get_ass_knowledge_emb():
    os.makedirs(os.path.join(text_dir, args.model_name), exist_ok=True)
    model, tokenizer = Load_Model_and_Tokenizer(args.model_name, args.device)
    model.to(torch.device(args.device))
    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')

    with open('text/ass09/id_skill_desc_dict.json','rb') as f:
        text = json.load(f)
    
            
    stu_all_info = {}
    print(len(text))
    for key, value in tqdm(text.items(), desc="Processing"):

        stu_emb_info = get_embed(model, tokenizer,value)
        stu_all_info[key] = stu_emb_info
            
    with open(os.path.join(text_dir, args.model_name)+'/knowledge_embedding.pkl','wb') as f:
        pickle.dump(stu_all_info, f) 
        
    



if __name__ == "__main__":
    

    embed = get_ass_stu_emb()
    # embed = get_ass_knowledge_emb()
    # embed = get_ass_two_emb()
    # embed = get_sum_emb()
    # get_full_text_emb_dict()
    # pipeline_my()
    # print(embed)
    