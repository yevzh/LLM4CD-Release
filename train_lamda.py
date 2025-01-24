import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dataset import Dataset
import json
import sys
from model import LamdaNet
from sklearn.metrics import roc_auc_score
import pickle
import os
from utils import *
import wandb

def infonce_loss(stu_emb, question_emb, similar_students, temperature=0.1):

    stu_emb = stu_emb / torch.norm(stu_emb, dim=1, keepdim=True) 
    stu_num, _ = stu_emb.shape
    losses = []

    for stu_id, sim_stu_ids in similar_students.items():
        stu_id = int(stu_id) - 1  
        sim_stu_ids = [int(i) - 1 for i in sim_stu_ids]

        stu_vec = stu_emb[stu_id]
        sim_stu_vecs = stu_emb[sim_stu_ids]

        pos_scores = torch.exp(torch.matmul(stu_vec, sim_stu_vecs.T) / temperature)


        all_indices = set(range(stu_num))
        neg_indices = list(all_indices - set(sim_stu_ids) - {stu_id})
        neg_stu_vecs = stu_emb[neg_indices]


        neg_scores = torch.exp(torch.matmul(stu_vec, neg_stu_vecs.T) / temperature)


        loss = -torch.log(torch.sum(pos_scores) / (torch.sum(pos_scores) + torch.sum(neg_scores)))
        losses.append(loss)


    return torch.mean(torch.stack(losses))

def train_lamda(args, TrainLoader, ValidLoader,TestLoader, student_n, exer_n, knowledge_n):
    if  args.dataset == 'boyu':
        with open(f'{args.text_dir}{args.llm}/question_embedding.pkl', 'rb') as fp:
            knowledge_text = pickle.load(fp)    
    else:
        with open(f'{args.text_dir}{args.llm}/knowledge_embedding.pkl', 'rb') as fp:
            knowledge_text = pickle.load(fp)
        with open(f'{args.text_dir}{args.llm}/question_embedding_{args.ques_emb}.pkl', 'rb') as fp:
            knowledge_text = pickle.load(fp)    
            

    with open(f'{args.text_dir}{args.llm}/student_embedding.pkl', 'rb') as fp:
        diagnosis_text = pickle.load(fp)

    with open(f'{args.text_dir}{args.llm}/student_embedding.pkl', 'rb') as fp:
        student_text = pickle.load(fp)

    print('finetuning model...')

    net = LamdaNet(knowledge_n, exer_n, student_n, args)

    # net.load_state_dict(torch.load('saved_models/tmp_ttt.pth'))
    net = net.to(args.device)
    relation_map = construct_relation_graph(args)
    best = 0
    best_model = net.state_dict()
    loss_function = nn.BCELoss()
    optimizer = optim.AdamW(net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    batch_count = 0
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        batch_count = 0
        for batch in TrainLoader:
            batch_count += 1
            input_stu_ids, input_exer_ids,  labels,input_knowledge_embs= batch
            input_kn_ids = np.argmax(input_knowledge_embs, axis=1)
            input_stu_ids, input_exer_ids,input_kn_ids, input_knowledge_embs, labels = input_stu_ids.to(args.device), input_exer_ids.to(args.device),input_kn_ids.to(args.device), input_knowledge_embs.to(args.device), labels.to(args.device)
            ques_emb = []
            stu_emb = []
            diag_emb = []
            for i in range(len(input_stu_ids)):
                ques_emb.append(knowledge_text[str(input_exer_ids[i].item()+1)])
            for i in range(len(input_stu_ids)):
                stu_emb.append(student_text[str(input_stu_ids[i].item()+1)])
            for i in range(len(input_stu_ids)):
                diag_emb.append(diagnosis_text[str(input_stu_ids[i].item()+1)])
            user_input = torch.tensor(np.array(stu_emb)).to(args.device)
            item_input = torch.tensor(np.array(ques_emb)).to(args.device)
            diag_input = torch.tensor(np.array(diag_emb)).to(args.device)
            user_input = user_input.float()
            item_input = item_input.float()
            diag_input = diag_input.float()
            optimizer.zero_grad()


            stu_output,_= net.forward(relation_map,user_input,diag_input, item_input,input_stu_ids, input_exer_ids)

            cd_loss = loss_function(stu_output, labels.float())
            

            loss = cd_loss
                
            loss.backward()
            optimizer.step()

        
        rmse, auc = validate_lamda(args, net, epoch, ValidLoader, knowledge_text, student_text, diagnosis_text,relation_map)
        rmse_, auc_ = validate_lamda2(args, net, epoch, TestLoader, knowledge_text, student_text, diagnosis_text,relation_map)
        if auc > best:
            # net.save_emb(epoch, relation_map)
            best = auc
            best_model = net.state_dict()
            torch.save(best_model,'saved_models/{}.pth'.format(args.saved_model))

    net.load_state_dict(torch.load('saved_models/{}.pth'.format(args.saved_model)))
    
    test_lamda(args.n_knowledge, net, args, knowledge_text, student_text, diagnosis_text,relation_map) 
def validate_lamda(args,model,epoch, ValidLoader,knowledge_text, student_text, diagnosis_text,relation_map):
    model.eval()
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0.0, 0.0
    pred_all = []
    label_all = []
    for batch in ValidLoader:
        batch_count+=1
        # print(batch_count)
        input_stu_ids, input_exer_ids,  labels,input_knowledge_embs= batch
        result_vector = np.argmax(input_knowledge_embs, axis=1)
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(args.device), input_exer_ids.to(args.device), input_knowledge_embs.to(args.device), labels.to(args.device)

        ques_emb = []
        stu_emb = []
        diag_emb = []
        for i in range(len(input_stu_ids)):
            ques_emb.append(knowledge_text[str(input_exer_ids[i].item()+1)])
        for i in range(len(input_stu_ids)):
            stu_emb.append(student_text[str(input_stu_ids[i].item()+1)])
        for i in range(len(input_stu_ids)):
            diag_emb.append(diagnosis_text[str(input_stu_ids[i].item()+1)])
        user_input = torch.tensor(np.array(stu_emb)).to(args.device)
        item_input = torch.tensor(np.array(ques_emb)).to(args.device)
        diag_input = torch.tensor(np.array(diag_emb)).to(args.device)
        user_input = user_input.float()
        item_input = item_input.float()
        diag_input = diag_input.float()

        output,_ = model.forward(relation_map,user_input,diag_input, item_input, input_stu_ids, input_exer_ids)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()        

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    label_all = label_all.astype(int)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('Valid: epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    if args.wandb:
        wandb.log({
        'Valid ACC':accuracy,
        'Valid AUC':auc,
        'Valid RMSE': rmse, 
        'Epoch': epoch+1
        })
    return rmse, auc
def validate_lamda2(args,model,epoch, ValidLoader,knowledge_text, student_text, diagnosis_text,relation_map):
    model.eval()
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0.0, 0.0
    pred_all = []
    label_all = []
    for batch in ValidLoader:
        batch_count+=1
        # print(batch_count)
        input_stu_ids, input_exer_ids,  labels,input_knowledge_embs= batch
        result_vector = np.argmax(input_knowledge_embs, axis=1)
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(args.device), input_exer_ids.to(args.device), input_knowledge_embs.to(args.device), labels.to(args.device)
        # result_tensor = np.dot(knowledge_tensor, embedding_tensor)
        
        # my_tensor = torch.tensor(result_tensor).to(args.device)
        ques_emb = []
        stu_emb = []
        diag_emb = []
        for i in range(len(input_stu_ids)):
            ques_emb.append(knowledge_text[str(input_exer_ids[i].item()+1)])
        for i in range(len(input_stu_ids)):
            stu_emb.append(student_text[str(input_stu_ids[i].item()+1)])
        for i in range(len(input_stu_ids)):
            diag_emb.append(diagnosis_text[str(input_stu_ids[i].item()+1)])
        user_input = torch.tensor(np.array(stu_emb)).to(args.device)
        item_input = torch.tensor(np.array(ques_emb)).to(args.device)
        diag_input = torch.tensor(np.array(diag_emb)).to(args.device)
        user_input = user_input.float()
        item_input = item_input.float()
        diag_input = diag_input.float()

        output,_ = model.forward(relation_map,user_input,diag_input, item_input, input_stu_ids, input_exer_ids)

        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()        

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    label_all = label_all.astype(int)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('Test: epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    if args.wandb:
        wandb.log({
        'Test ACC':accuracy,
        'Test AUC':auc,
        'Test RMSE': rmse, 
        'Epoch': epoch+1
        })
    return rmse, auc
def test_lamda(n_knowledge, model, args, knowledge_text, student_text, diagnosis_text,relation_map):
    TestDataset = Dataset(n_knowledge, 'test', args)
    TestLoader = DataLoader(TestDataset, batch_size = args.batch_size, shuffle=True)
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0.0, 0.0
    pred_all = []
    label_all = []
    for batch in TestLoader:
        batch_count+=1
        # print(batch_count)
        input_stu_ids, input_exer_ids,  labels,input_knowledge_embs= batch
        result_vector = np.argmax(input_knowledge_embs, axis=1)
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(args.device), input_exer_ids.to(args.device), input_knowledge_embs.to(args.device), labels.to(args.device)
        # result_tensor = np.dot(knowledge_tensor, embedding_tensor)
        
        # my_tensor = torch.tensor(result_tensor).to(args.device)
        ques_emb = []
        stu_emb = []
        diag_emb = []
        for i in range(len(input_stu_ids)):
            ques_emb.append(knowledge_text[str(input_exer_ids[i].item()+1)])
        for i in range(len(input_stu_ids)):
            stu_emb.append(student_text[str(input_stu_ids[i].item()+1)])
        for i in range(len(input_stu_ids)):
            diag_emb.append(diagnosis_text[str(input_stu_ids[i].item()+1)])
        user_input = torch.tensor(np.array(stu_emb)).to(args.device)
        item_input = torch.tensor(np.array(ques_emb)).to(args.device)
        diag_input = torch.tensor(np.array(diag_emb)).to(args.device)
        user_input = user_input.float()
        item_input = item_input.float()
        diag_input = diag_input.float()

        output,_ = model.forward(relation_map,user_input,diag_input, item_input, input_stu_ids, input_exer_ids)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()        

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    label_all = label_all.astype(int)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('Test accuracy= %f, rmse= %f, auc= %f' % (accuracy, rmse, auc))
    if args.wandb:
        wandb.log({
        'ACC':accuracy,
        'AUC':auc,
        'RMSE': rmse, 
        })    
    return rmse, auc

