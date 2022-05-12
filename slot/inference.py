# -*- coding: utf-8 -*-
# import jamspell
import json
from transformers import ElectraTokenizer,ElectraForTokenClassification
from utils import *
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import math
import time
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_PATH = "./model_trained/model_3_0.990"
MAX_LEN = 64
BATCH_SIZE = 16
NUM_LABEL = 26

class SlotClassify:
    def __init__(self,model_path,dict2label,device=None,max_len = MAX_LEN,batch_size = BATCH_SIZE):
        self.max_len = max_len
        self.batch_size = batch_size
        self.dict2label = self.get_json(dict2label)
        self.model_path = model_path
        if device!="cpu" and device!="cuda":
            self.device = torch.device(self.get_device())
        else:
            self.device = torch.device(device)
        print(self.device)
        self.load_model()
    
    def get_device(self):
        if torch.cuda.is_available():       
            device_name = "cuda"

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device_name = "cpu"
        return device_name

    def get_json(self,file_path):
        with open(file_path,'r') as f:
            res = json.load(f)
        return res

    def load_model(self):
        print('Loading BERT tokenizer...')
        self.tokenizer =  ElectraTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        self.model = ElectraForTokenClassification.from_pretrained(
                self.model_path,
                num_labels = NUM_LABEL,
                output_attentions = False,
                output_hidden_states = False,
        ).to(self.device)
        self.model.eval()

    def predict_text(self,texts):
        y_preds = []
        num_part = math.ceil(len(texts)/self.batch_size)
        y_preds = []
        for part in range(num_part):
            tokens =[]
            input_ids = []
            attention_masks = []
            b_texts = texts[part*self.batch_size:(part+1)*self.batch_size]
            marks = []
            index_mark = []
            for count,text in enumerate(b_texts):
                text_split = text.split()
                tokens.append(len(text_split))
                tmp_input_id = []
                mark =[]
                for idx,word in enumerate(text_split):
                    encode_id = self.tokenizer.encode(word)[1:-1]
                    tmp_input_id = tmp_input_id + encode_id
                    if len(encode_id)>1:
                        mark.append((idx,idx+len(encode_id)))
                tmp_input_id = tmp_input_id[0:MAX_LEN-2]
                tmp_input_id = [2] + tmp_input_id +[3]
                current_len = len(tmp_input_id)
                if current_len < MAX_LEN:
                    tmp_input_id += [0]*(MAX_LEN - current_len)
                att_mask = [int(token_id > 0) for token_id in tmp_input_id]

                assert len(tmp_input_id) == self.max_len == len(att_mask)

                input_ids.append(tmp_input_id)
                attention_masks.append(att_mask)
                if len(mark) > 0:
                    index_mark.append(count) # index of text
                    marks.append(mark) ## list of list of tuple
            train_inputs = input_ids
            train_masks= attention_masks
            train_inputs = torch.tensor(train_inputs,dtype=torch.long).to(self.device)
            train_masks = torch.tensor(train_masks,dtype=torch.long).to(self.device)

            output = self.model(train_inputs,attention_mask=train_masks)[0]
            
            output = torch.nn.functional.softmax(output,dim=-1)
            
            output = output.cpu().detach() ## batch_size x max_len x num_label
            indices = torch.argmax(output,dim=-1)
            indices = indices.tolist() # batch_size x max_len
            res = []
            assert len(indices) == len(tokens)
            for idx,value in enumerate(indices):
                b_res = []
                remove_index = []
                if idx in index_mark:
                    mark = marks[index_mark.index(idx)]
                    remove_index += [range(s+1,e) for s,e in mark]
                num_word = tokens[idx]
                for i in range(num_word):
                    if i not in remove_index:
                        b_res.append(value[i+1])
                res.append(b_res)

            y_preds += res
        return y_preds

    def get_label_name(self,lbs):
        res = []
        label_name = list(self.dict2label.keys())
        for lb in lbs:
            res.append(label_name[int(lb)])
        return res
    

def main():
    # test_path = "/home/ubuntu/truongnx/aihub/dataset/public_test_data/seq.in"
    # test_data = []
    # with open(test_path,'r') as lines:
    #     for line in lines:
    #         test_data.append(line.replace('\n',''))
    test_data = ['cho mình xem trạng thái của rgb 4 tại phòng gia đình 1']
    classifier = SlotClassify(MODEL_PATH,'../dataset/slot_label.json',device='cuda')
    preds = classifier.predict_text(test_data)
    # with open('result.txt','w+') as f:
    #     for i,pred in enumerate(preds):
    #         if len(pred) != len(test_data[i].split()):
    #             print(test_data[i],len(test_data[i].split()))
    #             print(pred,len(pred))
    #         lbs = classifier.get_label_name(pred)
    #         for lb in lbs:
    #             f.write(lb)
    #             f.write(' ')
    #         f.write('\n')
    
    print(preds)
if __name__ == "__main__":
    
    main()

