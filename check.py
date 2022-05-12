import pandas as pd

result = pd.read_csv('results.csv',header=None, usecols=[0,1])
old_result = pd.read_csv('old_results.csv',header=None, usecols=[0,1])
intents = result[0].values.tolist()
slots = result[1].values.tolist()
slots = [s.split() for s in slots]
old_intents = old_result[0].values.tolist()
old_slots = old_result[1].values.tolist()
old_slots = [s.split() for s in old_slots]
count_intent = 0
count_slot = 0
for i,_ in enumerate(intents):
    if old_intents[i]!=_:
        print(old_intents[i],"-"*10,_,"-"*10,i+1)
        count_intent +=1

print('{} changed intent'.format(count_intent))

for i,_ in enumerate(slots):
    if old_slots[i]!=_:
        print("#"*50)
        print('index: ',i+1)
        print('old: ',old_slots[i])
        print('new: ',_)
        count_slot +=1
print('{} changed slot'.format(count_slot))