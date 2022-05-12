import json
import os
from fix_label import fix_intent_label,fix_slot_label
from create_json import to_json

def data_dictionary(data_dir,filename,export_file_name = 'default'):
    list_text = []
    res = {}
    data_path = os.path.join(data_dir,filename)
    with open(data_path,'r') as lines:
        for line in lines:
            list_text.append(line.replace('\n',''))
    keys = list(dict.fromkeys(list_text).keys())
    assert len(keys) == len(set(keys))
    for i,k in enumerate(keys):
        res[k] = i
    if export_file_name == 'default':
        export_file_name = ".".join(filename.split('.')[:-1])+".json"
        export_path = os.path.join(data_dir,"..",export_file_name)
    with open(export_path,'w+') as f:
        json.dump(res,f)
    print('store successful!')

def read_data(file_path):
    res = []
    with open(file_path,'r') as lines:
        for line in lines:
            res.append(line.replace('\n',''))
    return res

def store_intent(path,data):
    with open(path,'w+') as f:
        for  ind,line in enumerate(data):
            f.write(str(line))           
            if ind<len(data)-1:               
                f.write('\n')
def store_slot(path,data):
    with open(path,'w+') as f:
        for ind,line in enumerate(data):
            for i,lb in enumerate(line):
                f.write(str(lb))
                if i<len(line):
                    f.write(' ')
            if ind<len(data)-1:               
                f.write('\n')

if __name__ == "__main__":
    # data_dir = "./dataset/training_data/"
    filename = "intent_label.json"
    filename2 = "slot_label.json"
    # data_dictionary(data_dir,filename2)

    prefix = "./dataset"
    list_path = ['training_data','dev_data']
    slot_lb_filename = 'seq.out'
    intent_lb_filename = 'label'
    input_filename = 'seq.in'

    with open("./dataset/"+filename,'r') as f:
        intent_dict = json.load(f)

    with open("./dataset/"+filename2,'r') as f:
        slot_dict = json.load(f)

    full_slot = []
    full_intent = []
    full_input = []

    
    for path in list_path:
        b_slot = []
        b_intent = []

        slot_path = os.path.join(prefix,path,slot_lb_filename)
        intent_path = os.path.join(prefix,path,intent_lb_filename)
        input_path = os.path.join(prefix,path,input_filename)

        slot_data = read_data(slot_path)
        intent_data = read_data(intent_path)
        input_data = read_data(input_path)

        ###convert slot data

        slot_label_fixed,set_slot_lb = fix_slot_label(slot_data,input_data)
        slot_dict = {key:str(i+2) for i,key in enumerate(list(set_slot_lb))}
        slot_dict['UNK'] = "1"
        slot_dict['PAD'] = "0"
        to_json(slot_dict,'./dataset/slot_label_2.json')

        for data in slot_label_fixed:
            lbs = data.split()
            new_lbs = [slot_dict[lb] for lb in lbs]
            b_slot.append(new_lbs)
            full_slot.append(new_lbs)
        
        ### convert intent data
       
        intent_label_fixed = fix_intent_label(intent_data,input_data)
        new_lbs = [intent_dict[lb] for lb in intent_label_fixed]
        b_intent = new_lbs
        full_intent += new_lbs

        ### join input
        full_input += input_data
        
        ### save part

        slot_store = os.path.join(prefix,path,'slot_lbs.txt')
        intent_store = os.path.join(prefix,path,'intent_lbs.txt')

        store_intent(intent_store,b_intent)
        store_slot(slot_store,b_slot)


    folder = ''

    slot_store = os.path.join(prefix,folder,'slot_lbs.txt')
    intent_store = os.path.join(prefix,folder,'intent_lbs.txt')
    input_store = './dataset/full_seq.in'

    
    store_intent(intent_store,full_intent)
    store_slot(slot_store,full_slot)
    
    ## store input
    with open(input_store,'w+') as f:
        for  ind,line in enumerate(full_input):
            f.write(str(line))           
            if ind<len(full_slot)-1:               
                f.write('\n')




        

