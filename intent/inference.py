# -*- coding: utf-8 -*-
# import jamspell
import json
from transformers import ElectraTokenizer,ElectraForSequenceClassification
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

MODEL_PATH = "./model_trained/1st_best"
MAX_LEN = 64
BATCH_SIZE = 16
NUM_LABEL = 9
class IntentClassify:
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

        self.model = ElectraForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels = NUM_LABEL,
                output_attentions = False,
                output_hidden_states = False,
        ).to(self.device)
        self.model.eval()

    def predict_text(self,texts):
        y_preds = []

        input_ids = []
        attention_masks = []
        num_part = math.ceil(len(texts)/self.batch_size)
        y_preds = []
        for part in range(num_part):
            input_ids = []
            attention_masks = []
            b_texts = texts[part*self.batch_size:(part+1)*self.batch_size]
            for text in b_texts:
                input_id = self.tokenizer.encode(text, max_length=self.max_len, truncation=True,pad_to_max_length=True)
                att_mask = [int(token_id > 0) for token_id in input_id]

                assert len(input_id) == self.max_len == len(att_mask)

                input_ids.append(input_id)
                attention_masks.append(att_mask)

            train_inputs = input_ids
            train_masks= attention_masks
            train_inputs = torch.tensor(train_inputs,dtype=torch.long).to(self.device)
            train_masks = torch.tensor(train_masks,dtype=torch.long).to(self.device)

            output = self.model(train_inputs,attention_mask=train_masks)[0]
            output = torch.nn.functional.softmax(output,dim=1)
            output = output.cpu().detach().numpy().tolist()
            b_preds = [np.argmax(b_output).item() for b_output in output]

            y_preds += b_preds

        return y_preds

    def get_label_name(self,lbs):
        res = []
        label_name = list(self.dict2label.keys())
        for lb in lbs:
            res.append(label_name[int(lb)])
        return res
    

def main():
    test_path = "/home/ubuntu/truongnx/aihub/dataset/public_test_data/seq.in"
    test_data = []
    with open(test_path,'r') as lines:
        for line in lines:
            test_data.append(line.replace('\n',''))

    classifier = IntentClassify(MODEL_PATH,'../dataset/intent_label.json',device='cuda')
    preds = classifier.predict_text(test_data)
    lbs = classifier.get_label_name(preds)
    print(len(lbs))
    with open('result.txt','w+') as f:
        for lb in lbs:
            f.write(lb)
            f.write('\n')

if __name__ == "__main__":
    
    main()

