# -*- coding: utf-8 -*-

import dgl
import torch
import networkx as nx

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_graph(args, type, node):
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(node)
    edge_list = []
    if type == 'direct':
        with open(f'{args.data_dir}graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        # g.add_edges(dst, src)
        return g
    elif type == 'undirect':
        with open(f'{args.data_dir}graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g
    elif type == 'k_from_e':
        with open(f'{args.data_dir}graph/k_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open(f'{args.data_dir}graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        with open(f'{args.data_dir}graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_u':
        with open(f'{args.data_dir}graph/e_from_u.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    
    
def construct_local_map(args):
    local_map = {
        'directed_g': build_graph(args, 'direct', args.n_knowledge),
        'undirected_g': build_graph(args, 'undirect', args.n_knowledge),
        'k_from_e': build_graph(args, 'k_from_e', args.n_knowledge + args.n_exer),
        'e_from_k': build_graph(args, 'e_from_k', args.n_knowledge + args.n_exer),
        'u_from_e': build_graph(args, 'u_from_e', args.n_stu + args.n_exer),
        'e_from_u': build_graph(args, 'e_from_u', args.n_stu + args.n_exer),
    }
    return local_map

def construct_relation_graph(args):
    local_map = {
        'directed_g': build_graph(args, 'direct', args.n_knowledge),
        'undirected_g': build_graph(args, 'undirect', args.n_knowledge),
        'k_from_e': build_graph(args, 'k_from_e', args.n_knowledge + args.n_exer),
        'e_from_k': build_graph(args, 'e_from_k', args.n_knowledge + args.n_exer),
        'u_from_e': build_graph(args, 'u_from_e', args.n_stu + args.n_exer),
        'e_from_u': build_graph(args, 'e_from_u', args.n_stu + args.n_exer),
    }
    return local_map


def compute_loss(embedding_1, embedding_2, input_ids, temperature):
    node_embedding_1 = F.normalize(embedding_1, dim=1)
    node_embedding_2 = F.normalize(embedding_2, dim=1)

    batch_node_embedding_1 = node_embedding_1[input_ids]
    batch_node_embedding_2 = node_embedding_2[input_ids]
    
    pos_sim_nodes = torch.sum(batch_node_embedding_1*batch_node_embedding_2, dim = -1)
    tot_sim_nodes = torch.matmul(batch_node_embedding_1, torch.transpose(node_embedding_2, 0, 1))
    ssl_logit = tot_sim_nodes-pos_sim_nodes[:, None]
    ssl_loss = torch.logsumexp(ssl_logit/temperature, dim=1)

    return ssl_loss.mean()