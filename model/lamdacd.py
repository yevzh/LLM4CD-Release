

import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
import warnings
import pickle
warnings.simplefilter("error")
from sklearn.exceptions import UndefinedMetricWarning
def warn(*args, **kwargs):
    pass
warnings.warn = warn

class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)


    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self,g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        # g.ndata['h'] = self.layernorm(g.ndata['h'])
        return g.ndata.pop('h')

class Fusion(nn.Module):
    def __init__(self, args):
        self.device = args.device
        self.knowledge_dim = args.n_knowledge
        self.exer_n = args.n_exer
        self.emb_num = args.n_stu
        self.stu_dim = self.knowledge_dim
        self.args = args
        if args.llm == 'bert':
            self.input_dim = 384
        elif args.llm == 'vicuna13b':
            self.input_dim = 5120
        else:
            self.input_dim = 4096

        super(Fusion, self).__init__()
        self.directed_gat = GraphLayer(self.input_dim, args.n_knowledge)
        self.undirected_gat = GraphLayer(self.input_dim, args.n_knowledge)
        self.k_from_e_gat = GraphLayer(self.input_dim, args.n_knowledge)
        self.e_from_k_gat = GraphLayer(self.input_dim, args.n_knowledge)
        if args.graph_layer == 2:
            self.k_from_e_gat2 = GraphLayer(args.n_knowledge, args.n_knowledge)
            self.e_from_k_gat2 = GraphLayer(args.n_knowledge, args.n_knowledge)
            self.relu = nn.ReLU()

    def forward(self, kn_emb, exer_emb, local_map):
        directed_g = local_map['directed_g'].to(self.device)
        undirected_g = local_map['undirected_g'].to(self.device)
        k_from_e = local_map['k_from_e'].to(self.device)
        e_from_k = local_map['e_from_k'].to(self.device)

        k_directed = self.directed_gat(directed_g, kn_emb)
        k_undirected = self.undirected_gat(undirected_g, kn_emb)
        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e_gat(k_from_e, e_k_graph)
        # 
        e_from_k_graph = self.e_from_k_gat(e_from_k, e_k_graph)
        # 
        if self.args.graph_layer == 2:
            k_from_e_graph = self.k_from_e_gat2(k_from_e, self.relu(k_from_e_graph))
            e_from_k_graph = self.e_from_k_gat2(e_from_k, self.relu(e_from_k_graph))
        A = kn_emb
        B = k_directed
        C = k_undirected
        D = k_from_e_graph[self.exer_n:]
        kn_emb = B+C+D
        A = exer_emb
        B = e_from_k_graph[0: self.exer_n]
        exer_emb = B
        

        

        return kn_emb, exer_emb
    

  
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        
class GraphEncoder(nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        # self.layers = clones(layer, N)
        self.layers = nn.ModuleList([
            Fusion(args),
            # FusionAdaptor(args)
        ])
        
    def forward(self, kn_emb, exer_emb, local_map):
        for layer in self.layers:
            kn_emb, exer_emb = layer(kn_emb, exer_emb, local_map)
        return kn_emb, exer_emb        

        # return kn_emb, exer_emb

class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, out_dim, input_dim,dropout):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_dim, out_dim),
                    nn.LeakyReLU(),
                    # nn.Dropout(p = dropout),
                    # nn.Linear(out_dim, out_dim),
                    # nn.LeakyReLU(),
                )

    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    """
    Mixture of Export
    """
    def __init__(self, exp_num, exp_dim, inp_dim, dropout):
        super(MoE, self).__init__()
        export_num, export_arch = exp_num, exp_dim
        self.export_num = export_num
        self.gate_net = nn.Linear(inp_dim, export_num)
        self.export_net = nn.ModuleList([MLP(export_arch, inp_dim, dropout) for _ in range(export_num)])
       
    def forward(self, x):
        gate = self.gate_net(x).view(-1, self.export_num)  # (bs, export_num)
        gate = nn.functional.softmax(gate, dim=-1).unsqueeze(dim=1) # (bs, 1, export_num)
        experts = [net(x) for net in self.export_net]
        experts = torch.stack(experts, dim=1)  # (bs, expert_num, emb)
        out = torch.matmul(gate, experts).squeeze(dim=1)
        return out
    

class predNet(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, out_dim, input_dim, dropout):
        super(predNet, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(p = dropout),
                    nn.Linear(512,1),
                    
                )

    def forward(self, x):
        return self.fc(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)

    
    def forward(self, x, seq_lens):
        batch_size, max_seq_len, emb_dim = x.size()
        
        q = self.query(x)
        k = self.key(x)
        # v = self.value(x)
        v = x
        
        q = q.view(batch_size, max_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, max_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, max_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(x.device) >= seq_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 1, -1e9)
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, max_seq_len, emb_dim)
        
        # output = self.fc(attn_output)
        output = attn_output
        
        mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(x.device) < seq_lens.unsqueeze(1)
        masked_output = output * mask.unsqueeze(-1)
        output = masked_output.sum(dim=1) / seq_lens.unsqueeze(-1)
        
        return output
    

class LamdaNet(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, args):
        self.knowledge_dim = knowledge_n
        self.moe_out_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.args = args
        if args.llm == 'bert':
            self.input_dim = 384
        elif args.llm == 'vicuna13b':
            self.input_dim = 5120
        else:
            self.input_dim = 4096
        # self.input_dim = 4096
        super(LamdaNet, self).__init__()
        # self.emdnet = nn.Sequential(nn.Linear(1, self.stu_dim), nn.ReLU(), nn.Linear(self.stu_dim, self.stu_dim))
        self.moe = MoE(5, self.moe_out_dim, self.input_dim, 0.2)
        self.stu_emb_pretrain, self.kn_emb_pretrain, self.exer_emb_pretrain = self.load_embeddings()
        self.stu_emb_pretrain = self.stu_emb_pretrain.to(args.device)
        # print(self.exer_emb_pretrain.shape)
        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.response_emb = nn.Embedding(2, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        
        


        self.mlp = nn.Sequential(
            nn.Linear(3*self.knowledge_dim, self.knowledge_dim),
            nn.ReLU()
        )
        self.pred_net = predNet(1,(self.prednet_input_len), 0.2 )

        
        with open(f'{self.args.data_dir}encode_info.json', 'r') as fp:
            self.encode_info = json.load(fp)
        with open(f'{self.args.data_dir}encode_score.json', 'r') as fp:
            self.encode_score = json.load(fp)
        with open(f'{self.args.data_dir}encode_kn.json', 'r') as fp:
            self.encode_kn = json.load(fp)

        self.graphencoder = GraphEncoder(args)
        self.exer_emb = nn.Embedding(self.exer_n,self.input_dim)
        self.kn_emb = nn.Embedding(self.knowledge_dim, self.input_dim)
        if args.freeze:
            self.kn_emb = nn.Embedding.from_pretrained(self.kn_emb_pretrain, freeze=True).to(self.args.device)
            self.exer_emb = nn.Embedding.from_pretrained(self.exer_emb_pretrain, freeze=True).to(self.args.device)
        else:
            self.kn_emb = nn.Embedding.from_pretrained(self.kn_emb_pretrain, freeze=False).to(self.args.device)
            self.exer_emb = nn.Embedding.from_pretrained(self.exer_emb_pretrain, freeze=False).to(self.args.device)

        for name, param in self.named_parameters():
            if 'weight' in name and not any(nd in name for nd in ['exer_emb','kn_emb']):
                if len(param.shape)<2:
                    nn.init.xavier_normal_(param.unsqueeze(0))
                else:
                    nn.init.xavier_normal_(param)
        self.alpha = nn.Parameter(torch.tensor(args.alpha))

    def get_state_representation(self, stu_ids, exer_emb, kn_emb):
        batch_size = stu_ids.size(0)
        max_seq_len = max(len(self.encode_info.get(str(stu_id.item() + 1), [])) for stu_id in stu_ids)

        question_ids_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(stu_ids.device)
        responses_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(stu_ids.device)
        kn_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(stu_ids.device)
        seq_lens = torch.zeros((batch_size,), dtype=torch.long).to(stu_ids.device)

        for i in range(batch_size):
            stu_id = str(stu_ids[i].item() + 1)
            question_ids = self.encode_info.get(stu_id, [])
            responses = self.encode_score.get(stu_id, [])
            
            kn_ids = self.encode_kn.get(stu_id, [])

            seq_len = len(question_ids)
            seq_lens[i] = seq_len
            question_ids_tensor[i, :seq_len] = torch.LongTensor(question_ids).to(stu_ids.device) - 1
            responses_tensor[i, :seq_len] = torch.LongTensor(responses).to(stu_ids.device)
            kn_tensor[i, :seq_len] = torch.LongTensor(kn_ids).to(stu_ids.device)

        question_embs = torch.sigmoid(exer_emb[question_ids_tensor])
        response_embs = torch.sigmoid(self.response_emb(responses_tensor))
        kn_embs = torch.sigmoid(kn_emb[kn_tensor])

        combined_embs = torch.cat((question_embs, kn_embs, response_embs), dim=2)

        if self.args.use_attention:
            state_reps = self.attention(combined_embs, seq_lens)
        else:
            mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(stu_ids.device) < seq_lens.unsqueeze(1)
            masked_combined_embs = combined_embs * mask.unsqueeze(-1)
            state_reps = masked_combined_embs.sum(dim=1) / seq_lens.unsqueeze(-1)

        return state_reps
    def forward(self,relation_map, stu_emb_raw, diag_emb_raw, ques_emb_raw, stu_id, exer_id):
        args = self.args
        # TO DO: here we use to process state encoder
                
        kn_emb = self.kn_emb(torch.arange(self.knowledge_dim).to(self.args.device))

        exer_emb = self.exer_emb(torch.arange(self.exer_n).to(self.args.device))


        kn_emb, exer_emb= self.graphencoder(kn_emb, exer_emb , relation_map)

        state_reps = self.get_state_representation(stu_id, exer_emb, kn_emb)        
        
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        stat_emb = stu_emb
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))

        stu_emb_after_moe = self.moe(stu_emb_raw)
        ques_emb_after_moe = self.moe(ques_emb_raw)
        

        stu_input = stu_emb_after_moe + self.mlp(state_reps)


        ques_input = ques_emb_after_moe + k_difficulty


        input_x = stu_input-ques_input

        output_1 = torch.sigmoid(self.pred_net(input_x))

        return output_1.view(-1), stu_input
    

    


    def load_embeddings(self):
        

        with open(f'{self.args.text_dir}{self.args.llm}/student_embedding.pkl', 'rb') as fp:
            embeddings_dict = pickle.load(fp)


        embedding_size = self.input_dim 
        num_embeddings = len(embeddings_dict)


        stu_embeddings_tensor = torch.zeros(num_embeddings, embedding_size)


        for key, embedding in embeddings_dict.items():
            index = int(key)
            
            stu_embeddings_tensor[index-1] = torch.tensor(embedding)
        
        with open(f'{self.args.text_dir}{self.args.llm}/knowledge_embedding.pkl', 'rb') as f:
            embeddings_dict = pickle.load(f)

        embedding_size = self.input_dim 
        num_embeddings = len(embeddings_dict)


        kn_embeddings_tensor = torch.zeros(num_embeddings, embedding_size)

        for key, embedding in embeddings_dict.items():
            index = int(key)
            kn_embeddings_tensor[index] = torch.tensor(embedding)

        if self.args.dataset == 'boyu' or self.args.dataset == 'new' or self.args.dataset == 'prob':
            with open(f'{self.args.text_dir}{self.args.llm}/question_embedding.pkl', 'rb') as f:
                embeddings_dict = pickle.load(f)

        else:
            with open(f'{self.args.text_dir}{self.args.llm}/question_embedding_{self.args.ques_emb}.pkl', 'rb') as f:
                embeddings_dict = pickle.load(f)

        embedding_size = self.input_dim 
        num_embeddings = len(embeddings_dict) 

        exer_embeddings_tensor = torch.zeros(num_embeddings, embedding_size)

        for key, embedding in embeddings_dict.items():
            index = int(key)

            exer_embeddings_tensor[index-1] = torch.tensor(embedding)

        return stu_embeddings_tensor, kn_embeddings_tensor, exer_embeddings_tensor  