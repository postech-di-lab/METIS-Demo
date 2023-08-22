from config import get_config
from data_loader import load_dataset
from preprocessed_dataset import load_multimodal_dataset
from METIS import METIS
import os

def main(args):
    # dataset load
    train_data, val_data, test_data, num_items = load_multimodal_dataset(args.dataset, args.repetitive)
    print("num_items", num_items)
    #exit()
    # Load learner 
    multimodal_option = {
        'image_weight': 0,
        'text_weight': 0,
    }

    dir_path = f'./logs/recent_log_{args.lr}_{args.dropout_rate}_{args.patience}_{multimodal_option["image_weight"]}_{multimodal_option["text_weight"]}'
    if not os.path.exists(dir_path): os.mkdir(dir_path)
    for idx in range(10):
        log_file_name = os.path.join(dir_path, f'{idx}.txt')
        with open(log_file_name, '+a') as log_file:    
            learner = METIS(args, num_items, multimodal_option=multimodal_option, log_file=log_file)
            # case1: train + validate + test
            learner.run(train_data, test_data)

    # log_file_name = f'./logs/recent_log_{args.lr}_{args.dropout_rate}_{multimodal_option["image_weight"]}_{multimodal_option["text_weight"]}.txt'
    # with open(log_file_name, '+a') as log_file:    
    #     learner = METIS(args, num_items, multimodal_option=multimodal_option, log_file=log_file)
    #     # case1: train + validate + test
    #     learner.run(train_data, val_data, test_data)

    # case2: train + validate(or test)
    # learner.run(train_data, val_data, None)
    # learner.run(train_data, None, test_data)

    # case3: only train
    # learner.run(train_data)

    # case4: validate or test
    # learner.validate(val_data)
    # learner.validate(test_data)
    
    # case5: predict
    # dis = learner.predict(test_data) # (29045, 33951) s x I
    # print("shape of dis", dis.shape)
    

if __name__ == "__main__":

    args = get_config()
    
    print("[Start]")
    print(args)

    main(args)

    print("[End]")
    print(args)