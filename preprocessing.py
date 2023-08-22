from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np

def split_index(train_sess_num, train_ratio=0.8, val_ratio=0.1):
    
    train_num = train_sess_num*train_ratio
    val_num = train_num + train_sess_num*val_ratio


    return train_num, val_num, train_sess_num

def filter_df_by_len(df, min_len, max_len):
    less_more = df.groupby('reviewerID').size()
    less_more_idx = less_more[(less_more >= min_len) & (less_more <= max_len)].index
    less_more_df = df[df['reviewerID'].isin(less_more_idx)]

    cut = df.groupby('reviewerID').size()
    cut_idx = cut[cut > max_len].index
    cut_df = df[df['reviewerID'].isin(cut_idx)].groupby('reviewerID').predicttail(50)

    filter_df = pd.concat([less_more_df, cut_df], axis=0)

    return filter_df

def preprocess(amazon_csv_path):
    arts_df = pd.read_csv(amazon_csv_path)
    #필요한 열만 사용
    arts_df = arts_df[['reviewTime', 'reviewerID', 'asin']]


    # review 3개 이상 한 사람의 세션 길이 구하려고
    test_df = arts_df.groupby('reviewerID').count()
    test_df = test_df[test_df['asin']>2]


    #session길이를 만드는 중
    test_df.rename(columns={'asin' : 'sessionlen'}, inplace=True)


    test_df.reset_index(drop=False)


    #세션안에서 시간순으로 정렬해주기위해 리뷰 시간을 datetime으로 바꿈
    arts_df['reviewTime'] = pd.to_datetime(arts_df['reviewTime'])


    df_sorted=arts_df.sort_values(by=['reviewerID','reviewTime'])


    df_merged = pd.merge(df_sorted, test_df, how='outer', on='reviewerID')


    df_merged = df_merged[df_merged['sessionlen']>2]
    df_merged = df_merged.reset_index(drop=True)


    #세션 아이티와 아이템 아이디를 붙여줌
    unique_arr=df_merged['reviewerID'].unique()
    unique_item=df_merged['asin'].unique()

    id_map={key:i for i, key in enumerate(unique_arr)}
    df_merged['sessionId'] = df_merged['reviewerID'].map(id_map)

    id_map={key:i for i, key in enumerate(unique_item)}
    df_merged['itemId'] = df_merged['asin'].map(id_map)


    df_merged.rename(columns={'reviewTime_x': 'reviewTime'}, inplace=True)


    df_merged = df_merged[['reviewTime', 'reviewerID', 'asin', 'sessionlen', 'sessionId', 'itemId']]


    test_df = df_merged.groupby('reviewerID').count()
    test_df = test_df[['sessionlen']]
    test_list = list(test_df['sessionlen'])


    val = 0
    len_list = []
    for i,idx in enumerate(test_list):
        for j in range(idx):
            len_list.append(val)
            val += 1
        val = 0




    df_merged['a'] = len_list


    df_merged['b'] = df_merged.a*timedelta(seconds=30)


    df_merged['datetime'] = df_merged['reviewTime'] + df_merged['b']


    df_merged['timestamp'] = pd.to_datetime(df_merged['datetime']).astype(np.int64) / 10**9


    df_merged = df_merged[['reviewerID', 'asin', 'itemId', 'sessionId', 'timestamp']]




    train_index,val_index,test_index = split_index(25000)


    df_train = df_merged[df_merged['sessionId'] < train_index]
    df_val = df_merged[(df_merged['sessionId'] >= train_index) & (df_merged['sessionId'] < val_index)]
    df_test = df_merged[(df_merged['sessionId'] >= val_index) & (df_merged['sessionId'] < test_index)]
    df_all = df_merged[df_merged['sessionId'] < test_index]




    min_len = 4
    max_len = 50
    train_filter = filter_df_by_len(df_train, min_len, max_len)
    val_filter = filter_df_by_len(df_val, min_len, max_len)
    test_filter = filter_df_by_len(df_test, min_len, max_len)
    return train_filter, val_filter, test_filter

def preprocess_and_save(amazon_csv_path, save_dir):
    train_df, val_df, test_df = preprocess(amazon_csv_path)
    save_dir = Path(save_dir)
    train_df.to_csv(save_dir/'train.csv')
    val_df.to_csv(save_dir/'valid.csv')
    test_df.to_csv(save_dir/'test.csv')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('csv_path',help='a CSV file path to preprocess.')
    parser.add_argument('save_dir',help='a directory to save preprocessed dataset files.')
    args = parser.parse_args()

    preprocess_and_save(args.csv_path, args.save_dir)