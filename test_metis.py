from config import get_config
from data_loader import load_dataset
from preprocessed_dataset import load_multimodal_dataset
from METIS import METIS

def run_multimodal():
    args = get_config({"device":"cuda:3","num_epoch":30})
    train_data, val_data, test_data, num_items = load_multimodal_dataset(args.dataset, args.repetitive)


    multimodal_option={"image_weight":0.0,"text_weight":0.0}
    with open("logs/multimodal_log.txt","+a") as log_file:
        metis=METIS(args,num_items=num_items,multimodal_option=multimodal_option, log_file=log_file)
        metis.run(train_data,val_data,test_data)

def run_plain():
    args = get_config({"device":"cuda:2", "num_epoch":30})
    train_data, val_data, test_data, num_items = load_dataset(args.dataset, args.repetitive)


    multimodal_option=None
    with open("logs/plain_log.txt","+a") as log_file:
        metis=METIS(args,num_items=num_items,multimodal_option=multimodal_option, log_file=log_file)
        metis.run(train_data,val_data,test_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--multimodal', action=argparse.BooleanOptionalAction)

    if parser.parse_args().multimodal:
        for _ in range(5):
            run_multimodal()
    else:
        for _ in range(5):
            run_plain()