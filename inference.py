# inference.py –data dataset
# Inference 결과를 좀 예쁘게 보여줘야함.
# - [] Print = (전체 성능 (train때 val 성능) + example)
# - [] Return = 각 사용자 별 추천 항목 상위 K개 담고있는 엑셀파일 출력?
# - [] 시각화?
# - [] 임의의 세션에 대한 출력? 길이가 1인 test set?
from datetime import datetime
from pathlib import Path

from config import get_parser
from preprocessed_dataset import load_multimodal_dataset
from METIS import METIS


def main(args):
    # dataset load
    train_data, val_data, test_data, num_items = load_multimodal_dataset(
        args.dataset, args.repetitive
    )
    print("num_items", num_items)
    # exit()
    # Load learner
    multimodal_option = {
        "image_weight": 0,
        "text_weight": 0,
    }

    if multimodal_option is None:
        dir_path = Path(f'./logs/recent_log_{args.lr}_{args.dropout_rate}')
    else:
        dir_path = Path(
            f'./logs/recent_log_{args.lr}_{args.dropout_rate}_{multimodal_option["image_weight"]}_{multimodal_option["text_weight"]}'
        )
    print(f"Opening log directory: {dir_path}")
    if not dir_path.exists():
        dir_path.mkdir()
    log_file_path = dir_path / f"{datetime.today().isoformat(timespec='seconds')}_inference.txt"
    with log_file_path.open("w") as log_file:
        learner = METIS(
            args, num_items, multimodal_option=multimodal_option, log_file=log_file
        )
        distances = learner.predict(test_data)  # -> session * Item?
    print(distances.shape)
    top_k = distances.topk(20, largest = False).indices - 1 # Predicted index starts at 1. However, ItemId starts at 0.
    result_path = dir_path / f"{datetime.today().isoformat(timespec='seconds')}_inference.csv"
    with result_path.open("w") as inference_result:
        for session_id, item_id_tensor in enumerate(top_k):
            inference_result.write(f"{session_id},{','.join(str(item_id.item()) for item_id in item_id_tensor)}\n")
    print(f"Saved the result in: {result_path}")
        


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("model_path", help="Path of model.")
    # parser.add_argument("--dataset", default="amazon_25000", help="Root of dataset folder. Must contain `train.csv`, `valid.csv`, `test.csv`, `image.safetensors`, and `text.safetensors`")
    main(parser.parse_args())
