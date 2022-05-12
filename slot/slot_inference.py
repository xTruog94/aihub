# -*- coding: utf-8 -*-

import json
from transformers import ElectraTokenizer,ElectraForTokenClassification,BertTokenizer,BertForTokenClassification
from .utils import *
import torch
import math


class SlotClassify:
    def __init__(self,model_path,dict2label,device,max_len,batch_size,num_label):
        self.max_len = max_len
        self.batch_size = batch_size
        self.dict2label = self.get_json(dict2label)
        self.model_path = model_path
        self.num_label = num_label

        if device!="cpu" and device!="cuda":
            self.device = torch.device(self.get_device())
        else:
            self.device = torch.device(device)
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
                num_labels = self.num_label,
                output_attentions = False,
                output_hidden_states = False,
        ).to(self.device)
        self.model.eval()

    def predict_text(self,texts):
        y_preds = []
        num_part = math.ceil(len(texts)/self.batch_size)
        y_preds = []
        for part in range(num_part):
            input_ids = []
            attention_masks = []
            b_texts = texts[part*self.batch_size:(part+1)*self.batch_size]
            marks = {}
            len_texts = []
            for count,text in enumerate(b_texts):
                text_split = text.split()
                len_texts.append(len(text_split) if len(text_split)<self.max_len else self.max_len )
                tmp_input_id = []
                mark = []
                for idx,word in enumerate(text_split):
                    encode_id = self.tokenizer.encode(word)[1:-1]
                    tmp_input_id = tmp_input_id + encode_id

                    if len(encode_id)>1:
                        tmp = list(range(idx+1,idx+len(encode_id))) ## +1 để không tính padding đầu câu
                        mark += tmp
                
                tmp_input_id = tmp_input_id[0:self.max_len-2]
                tmp_input_id = [2] + tmp_input_id +[3]
                current_len = len(tmp_input_id)
                if current_len < self.max_len:
                    tmp_input_id += [0]*(self.max_len - current_len)

                att_mask = [int(token_id > 0) for token_id in tmp_input_id]

                ### store text index has split word
                if len(mark) > 0 :
                    marks[str(count)] = mark
                assert len(tmp_input_id) == self.max_len == len(att_mask)

                # len_text = len(text_split) if len(text_split) <32 else 32
                # assert len(tmp_input_id)-len(mark) == len_text  
                input_ids.append(tmp_input_id)
                attention_masks.append(att_mask)

                    
            train_inputs = input_ids
            train_masks= attention_masks
            train_inputs = torch.tensor(train_inputs,dtype=torch.long).to(self.device)
            train_masks = torch.tensor(train_masks,dtype=torch.long).to(self.device)

            output = self.model(train_inputs,attention_mask=train_masks)[0]            
            output = torch.nn.functional.softmax(output,dim=-1)
            
            output = output.cpu().detach() ## batch_size x max_len x num_label
            indices = torch.argmax(output,dim=-1)
            indices = indices.tolist() # batch_size x max_len
            # print(indices)
            res = []
            assert len(indices) == len(b_texts)
            for idx,value in enumerate(indices):
                b_res = []
                index_token_remove = marks.get(str(idx),False)
                if index_token_remove :
                    b_res = [value[i] for i in range(len(value)) if i not in index_token_remove]
                else:
                    b_res = value
                # print('before',b_res)
                first_padding_index = b_res.index(0)
                last_padding_index = b_res.index(0,1)
                b_res = b_res[first_padding_index+1: last_padding_index]
                # print(16*part + idx + 1)
                # print('after',b_res)
                # print(len_texts[idx])
                assert len(b_res) == len_texts[idx]
                res.append(b_res)
            
            y_preds += res
        return y_preds

    def get_label_name(self,lbs):
        res = []
        label_name = {y: x for x, y in self.dict2label.items()}
        for lb in lbs:
            res.append(label_name[str(lb)])
        return res
    

def main():
    # test_path = "/home/ubuntu/truongnx/aihub/dataset/public_test_data/seq.in"
    # test_data = []
    # with open(test_path,'r') as lines:
    #     for line in lines:
    #         test_data.append(line.replace('\n',''))
    # classifier = SlotClassify(MODEL_PATH,'../dataset/slot_label.json',device='cuda')
    # preds = classifier.predict_text(test_data)
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
    slot_model_path = "./model_trained/model_7_0.994"
    slot_max_len = 64
    slot_num_label = 15
    slot_convert_lb_path = '../dataset/slot_label_2.json'
    BATCH_SIZE = 16
    device = 'cuda'
    test_data = ['tôi muốn tăng đèn cảnh 3']
    classifier = SlotClassify(slot_model_path,dict2label = slot_convert_lb_path,device=device,max_len=slot_max_len,batch_size=BATCH_SIZE,num_label=slot_num_label)
    preds = classifier.predict_text(test_data)
    print(preds)
if __name__ == "__main__":
    
    main()

