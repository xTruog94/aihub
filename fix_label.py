import json
import pandas as pd
from collections import Counter


def fix_intent_label(lbs,seq):
    res_lb = []
    for i,lb in enumerate(lbs):
        lb_split = lb.split('.')
        if lb_split[-1] == 'percentage' and 'phần trăm' not in seq[i] and 'mở' not in seq[i] and 'bật'not in seq[i] and 'đóng' not in seq[i] and 'tắt' not in seq[i]:
            # print(seq[i],i)
            lb_split[-1] = 'level'
        res_lb.append(".".join(lb_split))
    assert len(res_lb) == len(seq)
    return res_lb

def fix_slot_label(lbs,seq):
    res_lb = []
    set_lb = set()
    for i,lb in enumerate(lbs):
        sentence_split = seq[i].split()
        sentence_lb = []
        word_lb = lb.split()
        for w_lb in word_lb:
            if 'B-' in w_lb:
                tmp = w_lb.replace('B-','')
            elif 'I-' in w_lb:
                tmp = w_lb.replace('I-','')
            else:
                tmp = w_lb
            sentence_lb.append(tmp)
            set_lb.add(tmp)
        assert len(sentence_lb) == len(sentence_split)
        res_lb.append(" ".join(sentence_lb))
    assert len(res_lb) == len(seq)
    return res_lb,set_lb