import os
import pickle


def get_name(string):
    if r"resource/" in string:
        sub_string = string.split(r"resource/")[-1]
    elif r"property/" in string:
        sub_string = string.split(r"property/")[-1]
    else:
        sub_string = string.split(r"/")[-1]
    sub_string = sub_string.replace('_',' ')
    return sub_string

def read_structure_datas(data_path):
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
                print('loading a (id2object)file...    ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    def read_idobj_tuple_file(file_path):
        print('loading a idx_obj file...    ' + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip('\n').split('\t')
                ret.append( ( int(th[0]),th[1] ) )
        return ret
    print("load data from... :", data_path)
    #entity index(ent id)2entity, relation index(rel_id)2relation
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])
    index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])
    entity2index = {e:idx for idx,e in index2entity.items()}
    rel2index = {r:idx for idx,r in index2rel.items()}
    #relation triples
    rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
    rel_triples_2 = read_idtuple_file(data_path + 'triples_2')
    #entity_ill
    train_ill = read_idtuple_file(data_path + 'sup_pairs')
    test_ill = read_idtuple_file(data_path + 'ref_pairs')
    ent_ill = []
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)
    return ent_ill,index2rel, index2entity, rel2index, entity2index,rel_triples_1,rel_triples_2



def get_attribute_value_type(value,value_type):
    # return attributeValue type(integer/float/data/string)
    if value_type.endswith("<http://www.w3.org/2001/XMLSchema#integer>"):
        return "integer"
    elif value_type.endswith("<http://www.w3.org/2001/XMLSchema#double>"):
        return "float"
    elif value_type.endswith("<http://www.w3.org/2001/XMLSchema#date>"):
        return "date"
    elif value_type.endswith("<http://www.w3.org/2001/XMLSchema#gMonthDay>"):
        return "date"
    elif value_type == "string":
        return "string"
    else:
        try:
            int(value)
            return "integer"
        except:
            try:
                float(value)
                return "float"
            except:
                return "string"


def read_attribute_datas(kg1_att_file_name, kg2_att_file_name, entity_list, entity2index, add_name_as_attTriples = True):
    """
    return list of attribute triples [(entity_id,attribute,attributeValue,type of attributeValue)]
    """
    kg_att_datas = []
    with open(kg1_att_file_name,"r",encoding="utf-8") as f:
        for line in f:
            e, a, l, l_type = line.rstrip().split('\t')
            l_type = get_attribute_value_type(l,l_type)
            kg_att_datas.append((entity2index[e], a, l, l_type))
    with open(kg2_att_file_name,"r",encoding="utf-8") as f:
        for line in f:
            e, a, l, l_type = line.rstrip().split('\t')
            l_type = get_attribute_value_type(l, l_type)
            kg_att_datas.append((entity2index[e], a, l, l_type))
    if add_name_as_attTriples:
        for e in entity_list:
            l = get_name(e) #entity name
            a = 'name'
            l_type = 'string'
            kg_att_datas.append((entity2index[e], a, l, l_type))
    return kg_att_datas




