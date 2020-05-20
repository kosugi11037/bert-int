"""
Removing noise from attribute triples
example:
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      16
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      10
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      28
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      7
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      11
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      88
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      17
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      5
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      30
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      24
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      2
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      92
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      9
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      12
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      1
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      27
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      13
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      19
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      6
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      4
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      18
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      3
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      22
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      20
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      8
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      33
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      39
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      82
http://dbpedia.org/resource/ŠK_Slovan_Bratislava        no      29
"""
import os
import copy
from Param import *
from utils import fixed


def read_att_data(data_path):
    """
    load attribute triples file.
    """
    print("loading attribute triples file from: ",data_path)
    att_data = []
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            e,a,l = line.rstrip('\n').split(' ',2)
            e = e.strip('<>')
            a = a.strip('<>')
            if "/property/" in a:
                a = a.split(r'/property/')[-1]
            else:
                a = a.split(r'/')[-1]
            l = l.rstrip('@zhenjadefr .')
            if len(l.rsplit('^^',1)) == 2:
                l,l_type = l.rsplit("^^")
            else:
                l_type = 'string'
            l = l.strip("\"")
            att_data.append((e,a,l,l_type)) #(entity,attribute,value,value_type)
    return att_data



def file_make(keep_data,remove_data,keep_file_name,remove_file_name):
    """
    save
    """
    with open(keep_file_name,"w",encoding="utf-8") as f:
        for e,a,l,l_type in keep_data:
            string = e + '\t' + a + '\t' + l + '\t' + l_type + '\n'
            f.write(string)
    with open(remove_file_name,"w",encoding="utf-8") as f:
        for e,a,l,l_type in remove_data:
            string = e + '\t' + a + '\t' + l + '\t' + l_type + '\n'
            f.write(string)




def sort_a(data_list):
    """
    sort
    """
    new_data_list = []
    e2e_datas = dict()
    for e, a, l, l_type in data_list:
        if e not in e2e_datas:
            e2e_datas[e] = set()
        e2e_datas[e].add((e, a, l, l_type))
    for e in e2e_datas.keys():
        e2e_datas[e] = list(e2e_datas[e])
        e2e_datas[e].sort(key=lambda x: x[1])
        for one in e2e_datas[e]:
            new_data_list.append(one)
    return new_data_list



def remove_one_to_N_att_data_by_threshold(ori_keep_data,ori_remove_data,one2N_threshold):
    """
    Filter noise attribute triples based on threshold
    """
    att_data = copy.deepcopy(ori_keep_data)
    ori_remove_data = copy.deepcopy(ori_remove_data)
    e_a2fre = dict()
    for e,a,l,l_type in att_data:
        if (e,a) not in e_a2fre:
            e_a2fre[(e,a)] = 0
        e_a2fre[(e,a)] += 1
    remove_set = set()
    for e_a in e_a2fre:
        if e_a2fre[e_a] > one2N_threshold:
            remove_set.add(e_a)
    keep_datas = []
    remove_datas = []
    for e,a,l,l_type in att_data:
        if (e,a) in remove_set:
            remove_datas.append((e,a,l,l_type))
        else:
            keep_datas.append((e,a,l,l_type))

    keep_datas.sort(key=lambda x:x[0])
    remove_datas.sort(key=lambda x:x[0])
    keep_datas = sort_a(keep_datas)
    remove_datas = sort_a(remove_datas)
    print("Before removing noisy attribute triples, attribute triples {}".format(len(att_data)))
    remove_datas.extend(ori_remove_data)
    print("remaining attribute_triples num {} ; noisy attribute_triples num {}".format(len(keep_datas), len(remove_datas)))
    return keep_datas,remove_datas


if __name__ == '__main__':
    fixed(SEED_NUM)
    print("----------------clean attribute data--------------------")
    print("Start removing noise from attribute triples")
    #load attribute triples
    keep_data_1 = read_att_data(DATA_PATH + LANG + '_att_triples')
    keep_data_2 = read_att_data(DATA_PATH + 'en' + '_att_triples')

    remove_data_1 = []
    remove_data_2 = []

    #cleaned data save path:
    new_attribute_triple_1_file_path = DATA_PATH + 'new_att_triples_1'
    new_attribute_triple_2_file_path = DATA_PATH + 'new_att_triples_2'
    remove_attribute_triple_1_file_path = DATA_PATH + 'remove_att_triples_1'
    remove_attribute_triple_2_file_path = DATA_PATH + 'remove_att_triples_2'

    #clean data.
    keep_data_1, remove_data_1 = remove_one_to_N_att_data_by_threshold(keep_data_1,remove_data_1,one2N_threshold=3)
    keep_data_2, remove_data_2 = remove_one_to_N_att_data_by_threshold(keep_data_2,remove_data_2,one2N_threshold=3)

    remove_data_1 = sort_a(remove_data_1)
    remove_data_2 = sort_a(remove_data_2)
    keep_data_1 = sort_a(keep_data_1)
    keep_data_2 = sort_a(keep_data_2)

    #save.
    file_make(keep_data_1,remove_data_1,new_attribute_triple_1_file_path,remove_attribute_triple_1_file_path)
    file_make(keep_data_2,remove_data_2,new_attribute_triple_2_file_path,remove_attribute_triple_2_file_path)
