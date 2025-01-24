import torch
import logging as log

import os
import torch
import numpy as np
import pickle as pkl
import argparse
from fastchat.model import load_model, get_conversation_template, add_model_args
from sentence_transformers import SentenceTransformer
# from config import *
from typing import Dict, Tuple, Union, Optional
import json
import torch
from torch.nn import Module

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import *

def Load_Model_and_Tokenizer(model_name, device):

    if model_name == "bert":
        tokenizer = None
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        
    if model_name == "chatglm":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name], trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path[model_name], trust_remote_code=True).half().to(device)

    elif model_name == "chatglm2":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name], trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path[model_name], trust_remote_code=True).half().to(device) 
    elif model_name == "moss":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path[model_name], trust_remote_code=True).half().to(device)

    elif model_name in cn_alpaca_model_names:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path[model_name])
        model = LlamaForCausalLM.from_pretrained(model_path[model_name],
            load_in_8bit=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).half().to(device)
 
    elif model_name in belle_model_names:
        model = AutoModelForCausalLM.from_pretrained(model_path[model_name]).half().to(device)
        if model_name in belle_llama_model_names:
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path[model_name])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name])

    elif model_name == "InternLM-Chat-7B":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path[model_name], trust_remote_code=True).to(device)

    elif model_name in baichuan_model_names:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path[model_name], trust_remote_code=True).to(device)

    elif model_name in educhat_model_names:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path[model_name])
        model = LlamaForCausalLM.from_pretrained(model_path[model_name],torch_dtype=torch.float16,).half().to(device)

    elif model_name in wizard_model_names:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path[model_name])
        model = LlamaForCausalLM.from_pretrained(model_path[model_name],torch_dtype=torch.float16,).half().to(device)

    elif model_name in codet5p_model_names:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path[model_name],
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True,
                                              trust_remote_code=True).to(device)

    elif model_name == 'starchat':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[model_name])
    # to save memory consider using fp16 or bf16 by specifying torch_dtype=torch.float16 for example
        model = AutoModelForCausalLM.from_pretrained(model_path[model_name]).to(device)
    
    elif model_name == 'vicuna':
        model, tokenizer = load_model(model_path[model_name], "cuda", num_gpus=1, max_gpu_memory='40Gib')
    elif model_name in llama_model_names:
        tokenizer = LlamaTokenizer.from_pretrained(model_path[model_name])
        model = LlamaForCausalLM.from_pretrained(model_path[model_name]).to(device)
    elif model_name in codellama_model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_path[model_name])
        model = AutoModelForCausalLM.from_pretrained(model_path[model_name]).half().to(device)
        
    # 返回模型和分词器
    return model, tokenizer
    
def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data
def open_pkl(path_):
    with open(path_,'rb') as fh:
        data = pkl.load(fh)
    return data

def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    return data
def dump_pkl(path_, data):
    with open(path_, 'wb') as pkl_file:
        pkl.dump(data, pkl_file)
    return data

def batch_data_to_device(data, device):
    batch_x, y = data
    y = y.to(device)

    seq_num, x = batch_x
    seq_num = seq_num.to(device)
    x_len = len(x[0])
    # x_len = 8
    # log.info('x length {:d}'.format(x_len))
    for i in range(0, len(x)):
        for j in range(0, x_len):
            if isinstance(x[i][j], int):
                x[i][j] = torch.tensor(x[i][j]).to(device)
            else:
                x[i][j] = x[i][j].to(device)

    return [[seq_num, x], y]
