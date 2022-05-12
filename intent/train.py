import transformers
import pandas as pd
from transformers import AdamW,ElectraTokenizer,ElectraForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from arguments import load_args
from utils import *
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import time
import pickle
import os
from tqdm import tqdm
import datetime
# from DebertaTokenizer import debertaTokenizer
# from DebertaModel import Deberta
import shutil
def load_part_data(input_path,lb_path):
    texts = []
    labels = []
    with open(input_path, 'r') as lines:
        for line in lines:
            texts.append(line.replace('\n',''))

    with open(lb_path, 'r') as lines:
        for line in lines:
            labels.append(int(line.replace('\n','')))
    
    text_ids = []
    att_masks = []
    for text in tqdm(texts):
        # text_id, att_mask = tokenizer.encode(text, max_length=MAX_LEN, truncation=True,pad_to_max_length=True)
        text_id = tokenizer.encode(text, max_length=MAX_LEN, truncation=True,pad_to_max_length=True)
        att_mask = [int(token_id>0) for token_id in text_id]
        assert len(text_id) == len(att_mask)
        text_ids.append(text_id)
        att_masks.append(att_mask)

    
    train_x,train_y,train_m  = convert2torch(text_ids,labels,att_masks)
    
    train_data = TensorDataset(train_x, train_m, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    return train_dataloader

def load_data(input_path,lb_path):
    texts = []
    labels = []
    with open(input_path, 'r') as lines:
        for line in lines:
            texts.append(line.replace('\n',''))

    with open(lb_path, 'r') as lines:
        for line in lines:
            labels.append(int(line.replace('\n','')))
    
    text_ids = []
    att_masks = []
    for text in tqdm(texts):
        # text_id, att_mask = tokenizer.encode(text, max_length=MAX_LEN, truncation=True,pad_to_max_length=True)
        text_id = tokenizer.encode(text, max_length=MAX_LEN, truncation=True,pad_to_max_length=True)
        att_mask = [int(token_id>0) for token_id in text_id]
        assert len(text_id) == len(att_mask)
        text_ids.append(text_id)
        att_masks.append(att_mask)
    test_size = 0.2
    train_x, val_x, train_y, val_y = train_test_split(text_ids, labels, random_state=35, test_size=test_size)
    train_m, val_m = train_test_split(att_masks, random_state=35, test_size=test_size)
    # train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels, random_state=35, test_size=0.3)
    # train_m, test_val_m = train_test_split(att_masks, random_state=35, test_size=0.3)

    # test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=35, test_size=0.5)
    # test_m, val_m = train_test_split(test_val_m, random_state=35, test_size=0.5)
    
    train_x,val_x,train_y,val_y,train_m,val_m  = convert2torch(train_x,val_x,train_y,val_y,train_m,val_m)
    print(train_x.shape,val_x.shape,train_y.shape,val_y.shape,train_m.shape,val_m.shape)
    
    train_data = TensorDataset(train_x, train_m, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_x, val_m, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    # test_data = TensorDataset(test_x, test_m, test_y)
    # test_sampler = SequentialSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    return train_dataloader,val_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def trainer(train_dataloader,val_dataloader):
    print("start training")
    learning_rate = args.lr
    adam_epsilon = args.epsilon
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    num_epochs = args.epochs
    total_steps = len(train_dataloader) * num_epochs
    print(total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=total_steps)
    train_losses = []
    val_losses = []
    num_mb_train = len(train_dataloader)
    num_mb_val = len(val_dataloader)
    print(num_mb_train)
    if num_mb_val == 0:
        num_mb_val = 1
    epochs = num_epochs
    training_status = []
    best_acc = 0
    total_t0 = time.time()
    for epoch_i in range(0, num_epochs):

        #-------------------Training-----------------------
        print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

        t0 = time.time()
        total_loss = 0
        model.train()
        model.to(device)
        train_accuracy = 0
        nb_train_steps = 0
        for step, batch in enumerate(train_dataloader):
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
    
            outputs = model(input_ids=b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            predictions = outputs[1]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags  = b_labels.view(-1)
            tmp_train_accuracy = categorical_accuracy(predictions, tags, 0)
            train_accuracy += tmp_train_accuracy.item()
            nb_train_steps += 1
            avg_train_accuracy = train_accuracy/nb_train_steps
 
        avg_train_loss = total_loss / len(train_dataloader)             
        training_time = format_time(time.time() - t0)

        print("\n")
        print(" Average training loss: {0:.2f}".format(avg_train_loss))
        print(" Average training acc: {0:.2f}".format(avg_train_accuracy))
        print(" Training epoch took: {:}".format(training_time))
            
        # ------------------Validation--------------------
        print("\n")
        print("Validation")

        t0 = time.time()
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # model.to("cpu")
        # total_eval_accuracy = 0
        # total_eval_loss = 0
        # nb_eval_steps = 0

        for batch in val_dataloader:
        
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
                
            predictions = outputs[0]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags  = b_labels.view(-1)
            
            tmp_eval_accuracy = categorical_accuracy(predictions, tags, 0)
            eval_accuracy += tmp_eval_accuracy.item()
            nb_eval_steps += 1
            avg_val_accuracy = eval_accuracy/nb_eval_steps
        print("  Acc score: {0:.2f}".format(avg_val_accuracy))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

        if avg_val_accuracy >=  best_acc:

            output_dir = args.model_save + "model_" + str(epoch_i+1) + '_' + str(avg_val_accuracy)[:5]
            old_output_dir = args.model_save + "model_" + str(epoch_i) + '_' + str(best_acc)[:5]
            if os.path.exists(old_output_dir):
                shutil.rmtree(old_output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("Saving model to %s" % output_dir)

            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            best_acc = avg_val_accuracy
    print("\n")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


if __name__ == "__main__":

    args = load_args()
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    seed_val = 111
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    num_labels = args.max_labels
    cached_features_file = args.data_cache
    
    pre_trained = 'NlpHUST/electra-base-vn'

    tokenizer = ElectraTokenizer.from_pretrained(pre_trained,do_lower_case = False)
    model = ElectraForSequenceClassification.from_pretrained(pre_trained, num_labels=num_labels,
                                                                output_attentions=False, output_hidden_states=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    if os.path.exists(cached_features_file+"cache-train-{}-{}".format(args.max_len,args.batch_size)) and \
        os.path.exists(cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size)) and args.use_cache:
        # os.path.exists(cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size)) and \
        
        print("loading data train from cache...")
        train_dataloader = torch.load(cached_features_file+"cache-train-{}-{}".format(args.max_len,args.batch_size))
        val_dataloader = torch.load(cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size))
        # test_dataloader = torch.load(cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size))
    else:
        prefix = '../dataset/'
        input_file = prefix + 'full_seq.in'
        lb_path = prefix + 'intent_lbs.txt'
        training_dir = 'training_data'
        dev_dir = 'dev_data'
        if args.load_part:
            input_train_dir = os.path.join(prefix,training_dir,'seq.in')
            label_train_dir = os.path.join(prefix,training_dir,'intent_lbs.txt')
            input_val_dir = os.path.join(prefix,dev_dir,'seq.in')
            label_val_dir = os.path.join(prefix,dev_dir,'intent_lbs.txt')
            train_dataloader = load_part_data(input_train_dir,label_train_dir)
            val_dataloader = load_part_data(input_val_dir,label_val_dir)
        else:
            train_dataloader,val_dataloader = load_data(input_file,lb_path)
        torch.save(train_dataloader,cached_features_file+"cache-train-{}-{}".format(args.max_len,args.batch_size))
        torch.save(val_dataloader,cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size))
        # torch.save(test_dataloader,cached_features_file+"cache-test-{}-{}".format(args.max_len,args.batch_size))
    
    trainer(train_dataloader,val_dataloader)
    