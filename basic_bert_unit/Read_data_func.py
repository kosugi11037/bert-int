import pickle
from transformers import BertTokenizer
import logging
from Param import *
import pickle
import numpy as np
import re
import random
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)



def get_name(string):
    if r"resource/" in string:
        sub_string = string.split(r"resource/")[-1]
    elif r"property/" in string:
        sub_string = string.split(r"property/")[-1]
    else:
        sub_string = string.split(r"/")[-1]
    sub_string = sub_string.replace('_',' ')
    return sub_string



def ent2desTokens_generate(Tokenizer,des_dict_path,ent_list_1,ent_list_2,des_limit = DES_LIMIT_LENGTH - 2):
    #ent_list_1/2 == two different language ent list
    print("load desription data from... :", des_dict_path)
    ori_des_dict = pickle.load(open(des_dict_path,"rb"))
    ent2desTokens = dict()
    ent_set_1 = set(ent_list_1)
    ent_set_2 = set(ent_list_2)
    for ent,ori_des in ori_des_dict.items():
        if ent not in ent_set_1 and ent not in ent_set_2:
            continue
        string = ori_des
        encode_indexs = Tokenizer.encode(string)[:des_limit]
        ent2desTokens[ent] = encode_indexs
    print("The num of entity with description:",len(ent2desTokens.keys()))
    return ent2desTokens



def ent2Tokens_gene(Tokenizer,ent2desTokens,ent_list,index2entity,
                                ent_name_max_length = DES_LIMIT_LENGTH - 2):
    ent2tokenids = dict()
    for ent_id in ent_list:
        ent = index2entity[ent_id]
        if ent2desTokens!= None and ent in ent2desTokens:
            #if entity has description, use entity description
            token_ids = ent2desTokens[ent]
            ent2tokenids[ent_id] = token_ids
        else:
            #else, use entity name.
            ent_name = get_name(ent)
            token_ids = Tokenizer.encode(ent_name)[:ent_name_max_length]
            ent2tokenids[ent_id] = token_ids
    return ent2tokenids



def ent2bert_input(ent_ids,Tokenizer,ent2token_ids,des_max_length=DES_LIMIT_LENGTH):
    ent2data = dict()
    pad_id = Tokenizer.pad_token_id

    for ent_id in ent_ids:
        ent2data[ent_id] = [[],[]]
        ent_token_id = ent2token_ids[ent_id]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_id][0] = ent_token_ids
        ent2data[ent_id][1] = ent_mask_ids
    return ent2data






def read_data(data_path = DATA_PATH,des_dict_path = DES_DICT_PATH):
    def read_idtuple_file(file_path):
        print('loading a idtuple file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
        return ret
    def read_id2object(file_paths):
        id2object = {}
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                print('loading a (id2object)file...  ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    def read_idobj_tuple_file(file_path):
        print('loading a idx_obj file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip('\n').split('\t')
                ret.append( ( int(th[0]),th[1] ) )
        return ret

    print("load data from... :", data_path)
    #ent_index(ent_id)2entity / relation_index(rel_id)2relation
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])
    index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])
    entity2index = {e:idx for idx,e in index2entity.items()}
    rel2index = {r:idx for idx,r in index2rel.items()}

    #triples
    rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
    rel_triples_2 = read_idtuple_file(data_path + 'triples_2')
    index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')
    index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')

    #ill
    train_ill = read_idtuple_file(data_path + 'sup_pairs')
    test_ill = read_idtuple_file(data_path + 'ref_pairs')
    ent_ill = []
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)

    #ent_idx
    entid_1 = [entid for entid,_ in index_with_entity_1]
    entid_2 = [entid for entid,_ in index_with_entity_2]
    entids = list(range(len(index2entity)))

    #ent2descriptionTokens
    Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    if des_dict_path!= None:
        ent2desTokens = ent2desTokens_generate(Tokenizer,des_dict_path,[index2entity[id] for id in entid_1],[index2entity[id] for id in entid_2])
    else:
        ent2desTokens = None

    #ent2basicBertUnit_input.
    ent2tokenids = ent2Tokens_gene(Tokenizer,ent2desTokens,entids,index2entity)
    ent2data = ent2bert_input(entids,Tokenizer,ent2tokenids)

    return ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2



