from transformers import BertModel
import torch
import torch.nn as nn


class Basic_Bert_Unit_model(nn.Module):
    def __init__(self,input_size,result_size):
        super(Basic_Bert_Unit_model,self).__init__()
        self.result_size = result_size
        self.input_size = input_size
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.out_linear_layer = nn.Linear(self.input_size,self.result_size)
        self.dropout = nn.Dropout(p = 0.1)



    def forward(self,batch_word_list,attention_mask):
        x = self.bert_model(input_ids = batch_word_list,attention_mask = attention_mask)#token_type_ids =token_type_ids
        sequence_output, pooled_output = x
        cls_vec = sequence_output[:,0]
        output = self.dropout(cls_vec)
        output = self.out_linear_layer(output)
        return output

