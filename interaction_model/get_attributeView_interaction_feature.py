import os
import logging
logging.basicConfig(level=logging.ERROR)
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from read_data_func import *
from Param import *
from utils import *
from dual_aggregation_func import *





def attributeView_interaction_F_gene(ent_pairs, value_emb_list, ent2valueids, value_pad_id,
                                     kernel_num=21, cuda_num=0, batch_size=512):
    """
    Attribute-View Interaction.
    use Dual Aggregation and Attribute-View Interaction to generate Similarity Feature between entity pairs.
    return entity pairs and features(between entity pairs)
    """
    start_time = time.time()
    value_emb = torch.FloatTensor(value_emb_list).cuda(cuda_num)
    value_emb = F.normalize(value_emb, p=2, dim=-1)
    sigmas = kernel_sigmas(kernel_num).cuda(cuda_num)
    mus = kernel_mus(kernel_num).cuda(cuda_num)
    # print("sigmas:",sigmas)
    # print("mus:",mus)
    sigmas = sigmas.view(1,1,-1)
    mus = mus.view(1,1,-1)

    all_ent_pairs = []
    all_features = []
    # print("attributeValue embedding shape:", value_emb.shape)
    for start_pos in range(0, len(ent_pairs), batch_size):
        batch_ent_pairs = ent_pairs[start_pos: start_pos + batch_size]
        e1s = [e1 for e1, e2 in batch_ent_pairs]
        e2s = [e2 for e1, e2 in batch_ent_pairs]
        e1_values = [ent2valueids[e1] for e1 in e1s] #size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
        e2_values = [ent2valueids[e2] for e2 in e2s] #[B,ne2]

        e1_masks = np.ones(np.array(e1_values).shape)
        e2_masks = np.ones(np.array(e2_values).shape)
        e1_masks[np.array(e1_values) == value_pad_id] = 0
        e2_masks[np.array(e2_values) == value_pad_id] = 0
        e1_masks = torch.FloatTensor(e1_masks.tolist()).cuda(cuda_num).unsqueeze(-1)  # [B,ne1,1]
        e2_masks = torch.FloatTensor(e2_masks.tolist()).cuda(cuda_num).unsqueeze(-1)  # [B,ne2,1]

        e1_values = torch.LongTensor(e1_values).cuda(cuda_num)  # [B,ne1]
        e2_values = torch.LongTensor(e2_values).cuda(cuda_num)  # [B,ne2]
        e1_values_emb = value_emb[e1_values]  #[B,ne1,embedding_dim]
        e2_values_emb = value_emb[e2_values]  #[B,ne2,embedding_dim]

        sim_matrix = torch.bmm(e1_values_emb, torch.transpose(e2_values_emb, 1, 2))  # [B,ne1,ne2]
        features = batch_dual_aggregation_feature_gene(sim_matrix, mus, sigmas, e1_masks, e2_masks)
        features = features.detach().cpu().tolist()

        all_ent_pairs.extend(batch_ent_pairs)
        all_features.extend(features)

    print("all attribute-view interaction feature shape:", np.array(all_features).shape)
    print("get attribute-view interaction feature using time {:.3f}".format(time.time() - start_time))
    return all_ent_pairs, all_features


def main():
    print("----------------get attribute view interaction similarity feature--------------------")
    cuda_num = CUDA_NUM
    print("GPU NUM :", cuda_num)
    ent_ill, index2rel, index2entity, \
    rel2index, entity2index, \
    rel_triples_1, rel_triples_2 = read_structure_datas(DATA_PATH)

    # load attribute triples files.
    ents = [ent for ent in entity2index.keys()]
    new_attribute_triple_1_file_path = DATA_PATH + 'new_att_triples_1'
    new_attribute_triple_2_file_path = DATA_PATH + 'new_att_triples_2'
    print("loading attribute triples from: ", new_attribute_triple_1_file_path)
    print("loading attribute triples from: ", new_attribute_triple_2_file_path)
    att_datas = read_attribute_datas(new_attribute_triple_1_file_path,
                                     new_attribute_triple_2_file_path,
                                     ents, entity2index, add_name_as_attTriples=True)


    #value2index_dict generate
    value_emb = pickle.load(open(ATTRIBUTEVALUE_EMB_PATH, "rb"))
    value_list = pickle.load(open(ATTRIBUTEVALUE_LIST_PATH, "rb"))
    index2value = {v_id:value for v_id,value in enumerate(value_list)}
    value2index = {value:v_id for v_id,value in index2value.items()}


    #load ent_pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH,"rb"))

    # set <PAD>
    value_pad_id = len(value_emb)
    dim = len(value_emb[0])
    value_emb.append([0.0 for _ in range(dim)])
    index2value[value_pad_id] = '<PAD>'
    value2index['<PAD>'] = value_pad_id

    # ent2attribute_values (key = entity ,value = (padding) attribute_values of entity)
    entid_list = [eid for eid in index2entity.keys()]
    ent2values = ent2attributeValues_gene(entid_list,att_datas,max_length=ENTITY_ATTVALUE_MAX_NUM,pad_value='<PAD>')
    ent2valueids = dict()
    for e,values in ent2values.items():
        valueids = [value2index[v] for v in values]
        ent2valueids[e] = valueids


    # generate attribute-view interaction feature
    entity_pairs_1, features = attributeView_interaction_F_gene(entity_pairs, value_emb, ent2valueids, value_pad_id,
                                   kernel_num=KERNEL_NUM,cuda_num=cuda_num,batch_size=2048)

    for i in range(len(entity_pairs)):
        assert entity_pairs[i] == entity_pairs_1[i]

    # save
    pickle.dump(features, open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH, "wb"))
    print("save attribute-view similarity Feature in: ",ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH)

if __name__ == '__main__':
    fixed(SEED_NUM)
    main()