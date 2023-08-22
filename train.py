# train.py –data dataset
# - [x] Train같은 경우, hyper-parameter search 코드가 알아서 해야함.
# - [x] Save model
# - [x] Print performance
from itertools import product
from pathlib import Path
from datetime import datetime

from config import get_config, get_parser
from preprocessed_dataset import load_multimodal_dataset
from METIS import METIS


def main(args):
    # dataset load
    train_data, val_data, test_data, num_items = load_multimodal_dataset(args.dataset, args.repetitive)
    print("num_items", num_items)
    #exit()
    # Load learner 
    if args.text_weight is None and args.image_weight is None:
        multimodal_option = None
    else: 
        multimodal_option = {
            'image_weight': 0 if args.image_weight is None else args.image_weight,
            'text_weight': 0 if args.text_weight is None else args.text_weight,
        }
    
    model_save_dir = Path(args.model_save_dir)
    if not model_save_dir.exists():
        model_save_dir.mkdir()
    best_recall50 = 0
    best_lr = None
    best_dropout = None
    for lr, dropout in product(args.lrs, args.dropout_rates):
        args.lr = lr
        args.dropout_rate = dropout
        if multimodal_option is None:
            dir_path = Path(f'./logs/recent_log_{lr}_{dropout}')
        else:
            dir_path = Path(
                f'./logs/recent_log_{lr}_{dropout}_{multimodal_option["image_weight"]}_{multimodal_option["text_weight"]}'
            )
        if not dir_path.exists():
            dir_path.mkdir()
        log_file_path = dir_path / f"{datetime.today().isoformat(timespec='seconds')}.txt"
        with log_file_path.open('+a') as log_file:    
            learner = METIS(args, num_items, multimodal_option=multimodal_option, log_file=log_file, model_save_dir = model_save_dir)
            # case1: train + validate + test
            recall50 = learner.run(train_data, val_data, test_data)
            if best_recall50 < recall50:
                best_recall50 = recall50
                best_lr = lr
                best_dropout = dropout
    calibration = f"lr: {best_lr}   dropout:   {best_dropout}  recall@50:  {best_recall50}"
    print(calibration)
    with open("calibrations.txt", "+a") as calibration_log:
        calibration_log.write(f"\n{calibration}\n")
        calibration_log.write(f"args: {args}\n")


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("model_save_dir",nargs='?', default="./models", help="Directory that models will be saved in.")
    parser.add_argument("--text_weight", help="Sets a weight of text features. Range: [0.0,1.0]", type=float)
    parser.add_argument("--image_weight", help="Sets a weight of image features Range: [0.0,1.0]", type=float)
    parser.add_argument('--lrs', help="List of learning rates that will be tuned", type=float, default=[], action="extend", nargs="+")
    parser.add_argument('--dropout_rates', help="List of dropout rates that will be tuned", type=float, default=[], action="extend", nargs="+")
    args = parser.parse_args()
    if len(args.lrs) == 0:
        args.lrs = [args.lr]
    if len(args.dropout_rates) == 0:
        args.dropout_rates = [args.dropout_rate] 
    print(args)
    main(args)