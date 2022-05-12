from slot import SlotClassify
from intent import IntentClassify
import pandas as pd
import os
import json

slot_model_path = "./slot/model_trained/model_8_0.992"
slot_max_len = 64
slot_num_label = 15
slot_convert_lb_path = './dataset/slot_label.json'

intent_model_path = './intent/model_trained/model_8_0.983'
intent_max_len = 64
intent_num_label = 10
intent_convert_lb_path = './dataset/intent_label.json'
BATCH_SIZE = 16
device = 'cuda'


intent_classifier = IntentClassify(intent_model_path,dict2label = intent_convert_lb_path,device=device,max_len=intent_max_len,batch_size=BATCH_SIZE,num_label=intent_num_label)

slot_classifier = SlotClassify(slot_model_path,dict2label = slot_convert_lb_path,device=device,max_len=slot_max_len,batch_size=BATCH_SIZE,num_label=slot_num_label)

def run_intent(test_data):
    preds = intent_classifier.predict_text(test_data)
    lbs = intent_classifier.get_label_name(preds)
    return lbs

def rollback_lb(lbs):

    res_lbs = []
    
    for lb in lbs:
        res_lb = []
        for i,sub_lb in enumerate(lb):
            if sub_lb == 'O':
                tmp = 'O'
            else:
                if i == 0:
                    tmp = 'B-'+ sub_lb
                else:
                    before_sub_lb = lb[i-1]
                    if before_sub_lb == sub_lb:
                        tmp = 'I-'+ sub_lb
                    else:
                        tmp = 'B-'+ sub_lb
            res_lb.append(tmp)
        assert len(res_lb) == len(lb)
        res_lbs.append(res_lb)
    assert len(res_lbs) == len(lbs)
    return res_lbs

def run_slot(test_data):
    preds = slot_classifier.predict_text(test_data)
    res = []
    for i,pred in enumerate(preds):
        lbs = slot_classifier.get_label_name(pred)
        res.append(lbs)
    res = rollback_lb(res)
    return res

def get_index_duplicate(list_slot):
    tuple_element = [(list_slot[i],list_slot[i+1])for i in range(len(list_slot)-1)]
    res = [tuple_element.index(e)+1 for e in tuple_element if e[0] == e[1] and e[0][0]=='B']
    return res

if __name__ == "__main__":
    test_path = "/home/ubuntu/truongnx/aihub/dataset/public_test_data/seq.in"
    test_data = []
    with open(test_path,'r') as lines:
        for line in lines:
            test_data.append(line.replace('\n',''))
    
    intent_out = run_intent(test_data)
    slot_out = run_slot(test_data)
    assert len(intent_out) == len(slot_out)

    slot_out_processed = [" ".join(t) for t in slot_out]
    df = pd.DataFrame(data={'intent':intent_out,'slot':slot_out_processed})

    if os.path.exists('old_results.csv') and os.path.exists('results.csv'):
        os.remove('old_results.csv')
        os.rename('results.csv','old_results.csv')
    df.to_csv('results.csv',index=False,header=False)
