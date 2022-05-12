import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained',default='NlpHUST/electra-base-vn')
    parser.add_argument('--model_save',default='./model_trained/')
    parser.add_argument('--data_cache',default="./cache/")
    parser.add_argument('--max_len',default=64,type=int)
    parser.add_argument('--max_labels',default=15,type=int)
    parser.add_argument('--gpu_id',default=0,type=int)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--epochs',default=8,type=int)
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--epsilon',default=3e-7,type=float)
    parser.add_argument('--use_cache',action='store_true')
    parser.add_argument('--load_part',action='store_true')
    args = parser.parse_args()
    return args
