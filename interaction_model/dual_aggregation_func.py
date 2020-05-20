import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def kernel_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return l_mu
    bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return torch.FloatTensor(l_mu)

def kernel_sigmas(n_kernels):
    l_sigma = [0.001]  # for exact match.
    # small variance -> exact match
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return torch.FloatTensor(l_sigma)


def batch_dual_aggregation_feature_gene(batch_sim_matrix,mus,sigmas,attn_ne1,attn_ne2):
    """
    Dual Aggregation.
    [similarity matrix -> feature]
    :param batch_sim_matrix: [B,ne1,ne2]
    :param mus: [1,1,k(kernel_num)]
    :param sigmas: [1,1,k]
    :param attn_ne1: [B,ne1,1]
    :param attn_ne2: [B,ne2,1]
    :return feature: [B,kernel_num * 2].
    """
    sim_maxpooing_1, _ = batch_sim_matrix.topk(k=1,dim=-1)#[B,ne1,1] #get max value.
    pooling_value_1 = torch.exp((- ((sim_maxpooing_1 - mus) ** 2)/ (sigmas ** 2) / 2))  #[B,ne1,k]
    log_pooling_sum_1 = torch.log(torch.clamp(pooling_value_1, min=1e-10)) * attn_ne1 * 0.01 #[B,ne1,k]
    log_pooling_sum_1 = torch.sum(log_pooling_sum_1, 1)  # [B,k]

    sim_maxpooing_2,_ = torch.transpose(batch_sim_matrix, 1, 2).topk(k=1,dim=-1)#[B,ne2,1]
    pooling_value_2 = torch.exp((- ((sim_maxpooing_2 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne2,k]
    log_pooling_sum_2 = torch.log(torch.clamp(pooling_value_2, min=1e-10)) * attn_ne2 * 0.01  # [B,ne2,k]
    log_pooling_sum_2 = torch.sum(log_pooling_sum_2, 1)  # [B,k]

    batch_ne2_num = attn_ne2.sum(dim = 1)#[B,1]
    batch_ne2_num = torch.clamp(batch_ne2_num, min=1e-10)
    log_pooling_sum_2 = log_pooling_sum_2 * (1 / batch_ne2_num )#[B,k]

    batch_ne1_num = attn_ne1.sum(dim = 1)#[B,1]
    batch_ne1_num = torch.clamp(batch_ne1_num, min=1e-10)
    log_pooling_sum_1 = log_pooling_sum_1 * (1 / batch_ne1_num)  # [B,k]
    return torch.cat([log_pooling_sum_1,log_pooling_sum_2],dim=-1)

