import json

def to_json(data,file_path):
    with open(file_path,'w+')as f:
        json.dump(data,f)

# slot_label = {}
# with open("./dataset/training_data/slot_label.txt",'r') as lines:
#     for i,line in enumerate(lines):
#         slot_label[line.replace('\n','')] = str(i)
        
# to_json(slot_label,'./dataset/slot_label.json')

# intent_label = {}
# with open("./dataset/training_data/intent_label.txt",'r') as lines:
#     for i,line in enumerate(lines):
#         intent_label[line.replace('\n','')] = str(i)

# to_json(intent_label,'./dataset/intent_label.json')