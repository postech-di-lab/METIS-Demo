from collections import namedtuple
import argparse

import torch

def get_config(arg_dict={}):
    default_args = {
        "device" : "cuda:1" if torch.cuda.is_available() else 'cpu',
        "lr" : 0.001, # 0.0002
        "dropout_rate" : 0.0,
        "patience" : 10,
        # "device" : 'cpu',
        #"dataset" : "diginetica",
        "dataset" : "amazon_25000",
        #"batch_size" : 64,
        "batch_size" : 32,
        "val_batch_size" : 256,
        # "batch_size" : 16,
        # "val_batch_size" : 256,
        "embed_dim" : 128,
        "k" : 300,
        "margin" : 0.1,
        "lambda_dist" : 0.2,
        "lambda_orthog" : 0.0,
        "E" : 10,
        "max_position" : 50,
        "t0" : 3.0,
        "te" : 0.01,
        "num_epoch" : 300, #500,
        "repetitive" : False
    }
    Args = namedtuple("Args", default_args.keys())
    for key in default_args:
        if key in arg_dict:
            default_args[key] = arg_dict[key]
    args = Args(**default_args)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--device', default = "cuda" if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--dataset', default='diginetica')
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--val_batch_size', type=int, default=512)
    # parser.add_argument('--embed_dim', type=int, default=128)
    # parser.add_argument('--lr', type=float, default=0.0002)
    # parser.add_argument('--k', type=int, default=300) # the number of proxy
    # parser.add_argument('--dropout_rate', type=float, default=0.0)
    # parser.add_argument('--margin', type=float, default=0.1)
    # parser.add_argument('--lambda_dist', type=float, default=0.2)
    # parser.add_argument('--lambda_orthog', type=float, default=0.0)
    # parser.add_argument('--E', type=int, default=10)
    # parser.add_argument('--patience', type=int, default=10)
    # parser.add_argument('--max_position', type=int, default=50)
    # parser.add_argument('--t0', type=float, default=3.0)
    # parser.add_argument('--te', type=float, default=0.01)
    # parser.add_argument('--num_epoch', type=int, default=500)
    # parser.add_argument('--repetitive', type=bool, default=False)
    # args = parser.parse_args()
    return args

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dataset', default="amazon_25000")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=300) # the number of proxy
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--lambda_dist', type=float, default=0.2)
    parser.add_argument('--lambda_orthog', type=float, default=0.0)
    parser.add_argument('--E', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_position', type=int, default=50)
    parser.add_argument('--t0', type=float, default=3.0)
    parser.add_argument('--te', type=float, default=0.01)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--repetitive', type=bool, default=False)
    return parser