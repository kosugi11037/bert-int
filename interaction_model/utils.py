import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import re
import numpy as np
import copy
import random

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def neigh_ent_dict_gene(rel_triples,max_length,pad_id = None):
    """
    get one hop neighbor of entity
    return a dict, key = entity, value = (padding) neighbors of entity
    """
    neigh_ent_dict = dict()
    for h,r,t in rel_triples:
        if h not in neigh_ent_dict:
            neigh_ent_dict[h] = []
        if t not in neigh_ent_dict:
            neigh_ent_dict[t] = []
    for h,r,t in rel_triples:
        if h == t:
            continue
        neigh_ent_dict[h].append(t)
        neigh_ent_dict[t].append(h)
    #In order to get the maximum number of neighbors randomly for each entity
    for e in neigh_ent_dict.keys():
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
    for e in neigh_ent_dict.keys():
        neigh_ent_dict[e] = neigh_ent_dict[e][:max_length]
    if pad_id != None:
        for e in neigh_ent_dict.keys():
            pad_list = [pad_id] * (max_length - len(neigh_ent_dict[e]))
            neigh_ent_dict[e] = neigh_ent_dict[e] + pad_list
    return neigh_ent_dict




def ent2attributeValues_gene(entid_list,att_datas,max_length,pad_value = None):
    """
    get attribute Values of entity
    return a dict, key = entity ,value = (padding) attribute_values of entity
    """
    ent2attributevalues = dict()
    for e in entid_list:
        ent2attributevalues[e] = []
    for e,a,l,l_type in att_datas:
        ent2attributevalues[e].append(l)
    # random choose attributeValue to maxlength.
    for e in ent2attributevalues.keys():
        np.random.shuffle(ent2attributevalues[e])
    for e in ent2attributevalues.keys():
        ent2attributevalues[e] = ent2attributevalues[e][:max_length]
    if pad_value != None:
        for e in ent2attributevalues.keys():
            pad_list = [pad_value] * (max_length - len(ent2attributevalues[e]))
            ent2attributevalues[e] = ent2attributevalues[e] + pad_list
    return ent2attributevalues


def cos_sim_mat_generate(emb1,emb2,bs = 128,cuda_num = 0):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2,dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2,dim=1)
    res_mat = batch_mat_mm(array_emb1,array_emb2.t(),cuda_num,bs=bs)
    return res_mat


def batch_mat_mm(mat1,mat2,cuda_num,bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].cuda(cuda_num)
        res = temp_div_mat_1.mm(mat2.cuda(cuda_num))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat




def batch_topk(mat,bs=128,topn = 50,largest = False,cuda_num = 0):
    #be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat = mat[i:min(i+bs,axis_0)].cuda(cuda_num)
        score_mat,index_mat =temp_div_mat.topk(topn,largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score,0)
    res_index = torch.cat(res_index,0)
    return res_score,res_index




def test_topk_res(index_mat):
    ent1_num,ent2_num = index_mat.shape
    topk_list = [0 for _ in range(ent2_num)]
    MRR = 0
    for i in range(ent1_num):
        for j in range(ent2_num):
            if index_mat[i][j].item() == i:
                MRR += ( 1 / (j+1) )
                for h in range(j,ent2_num):
                    topk_list[h]+=1
                break
    topk_list = [round(x/ent1_num,5) for x in topk_list]
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_list[1 - 1], topk_list[10 - 1]), end="")
    if ent2_num >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_list[25 - 1]), end="")
    if ent2_num >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_list[50 - 1]), end="")
    print("")
    MRR/=ent1_num
    print("MRR:{:.5f}".format(MRR))









