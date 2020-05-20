import torch
import torch.nn as nn
import torch.nn.functional as F
from Param import *
import numpy as np
import time
import pickle
from eval_function import cos_sim_mat_generate,batch_topk,hit_res

def entlist2emb(Model,entids,entid2data,cuda_num):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb


def generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                entid2data,index2entity,
                                nearest_sample_num = NEAREST_SAMPLE_NUM, batch_size = CANDIDATE_GENERATOR_BATCH_SIZE):
    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        #langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0,len(train_ent1s),batch_size):
            temp_emb = entlist2emb(Model,train_ent1s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0,len(for_candidate_ent2s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent2s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        #language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0,len(train_ent2s),batch_size):
            temp_emb = entlist2emb(Model,train_ent2s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0,len(for_candidate_ent1s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent1s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        #cos sim
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1,for_candidate_emb1)
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2,for_candidate_emb2)
        torch.cuda.empty_cache()
        #topk index
        _,topk_index_1 = batch_topk(cos_sim_mat1,topn=nearest_sample_num,largest=True)
        topk_index_1 = topk_index_1.tolist()
        _,topk_index_2 = batch_topk(cos_sim_mat2,topn=nearest_sample_num,largest=True)
        topk_index_2 = topk_index_2.tolist()
        #get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)

        #show
        # def rstr(string):
        #     return string.split(r'/resource/')[-1]
        # for e in train_ent1s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
        # for e in train_ent2s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
    print("get candidate using time: {:.3f}".format(time.time()-start_time))
    torch.cuda.empty_cache()
    return candidate_dict






def train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,entid2data):
    print("start training...")
    for epoch in range(EPOCH_NUM):
        print("+++++++++++")
        print("Epoch: ",epoch)
        print("+++++++++++")
        #generate candidate_dict
        #(candidate_dict is used to generate negative example for train_ILL)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2
        print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
              .format(len(train_ent1s),len(train_ent2s),len(for_candidate_ent1s),len(for_candidate_ent2s)))
        candidate_dict = generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,
                                                     for_candidate_ent2s,entid2data,Train_gene.index2entity)
        Train_gene.train_index_gene(candidate_dict) #generate training data with candidate_dict

        #train
        epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data)
        Optimizer.zero_grad()
        torch.cuda.empty_cache()
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        if epoch >= 0:
            if epoch !=0:
                save(Model,train_ill,test_ill,entid2data,epoch)
            # test(Model,train_ill,entid2data,TEST_BATCH_SIZE,context="EVAL IN TRAIN SET")
            test(Model, test_ill, entid2data, TEST_BATCH_SIZE, context="EVAL IN TEST SET:")


def test(Model,ent_ill,entid2data,batch_size,context = ""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data,CUDA_NUM).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data,CUDA_NUM).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb1,emb2,batch_size,cuda_num=CUDA_NUM)
        score,top_index = batch_topk(res_mat,batch_size,topn = TOPK,largest=True,cuda_num=CUDA_NUM)
        hit_res(top_index)
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")


def save(Model,train_ill,test_ill,entid2data,epoch_num):
    print("Model {} save in: ".format(epoch_num), MODEL_SAVE_PATH + MODEL_SAVE_PREFIX + "model_epoch_" + str(epoch_num) + '.p')
    Model.eval()
    torch.save(Model.state_dict(),MODEL_SAVE_PATH + MODEL_SAVE_PREFIX + "model_epoch_" + str(epoch_num) + '.p')
    other_data = [train_ill,test_ill,entid2data]
    pickle.dump(other_data,open(MODEL_SAVE_PATH + MODEL_SAVE_PREFIX + 'other_data.pkl',"wb"))
    print("Model {} save end.".format(epoch_num))



def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s,pe2s,ne1s,ne2s in Train_gene:
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,entid2data,cuda_num=CUDA_NUM)
        pos_emb2 = entlist2emb(Model,pe2s,entid2data,cuda_num=CUDA_NUM)
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,entid2data,cuda_num=CUDA_NUM)
        neg_emb2 = entlist2emb(Model,ne2s,entid2data,cuda_num=CUDA_NUM)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM) #pos_score < neg_score
        batch_loss = Criterion( pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        batch_loss.backward()
        Optimizer.step()

        all_loss += batch_loss.item() * batch_length
    all_using_time = time.time()-start_time
    return all_loss,all_using_time









