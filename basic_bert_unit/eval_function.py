import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

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


def hit_res(index_mat):
    ent1_num,ent2_num = index_mat.shape
    topk_n = [0 for _ in range(ent2_num)]
    for i in range(ent1_num):
        for j in range(ent2_num):
            if index_mat[i][j].item() == i:
                for h in range(j,ent2_num):
                    topk_n[h]+=1
                break
    topk_n = [round(x/ent1_num,5) for x in topk_n]
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_n[1 - 1],topk_n[10 - 1]),end="")
    if ent2_num >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_n[25 - 1]),end="")
    if ent2_num >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_n[50 - 1]),end="")
    print("")





