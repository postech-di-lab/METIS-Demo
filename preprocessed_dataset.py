from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.models import convnext_tiny
from transformers import AutoTokenizer, DistilBertModel
from PIL import Image
from safetensors.torch import load_file, save_file


def preprocess_text(csv_path, device="cpu"):
    text_extractor = (
        DistilBertModel.from_pretrained("distilbert-base-uncased").to(device).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    csv_path = Path(csv_path)
    text_df = pd.read_csv(csv_path).set_index("asin")
    if "title" in text_df and "description" in text_df:
        text_series = text_df.apply(
        lambda row: row.title + ("" if pd.isna(row.description) else row.description),
        axis=1,
    )
    elif "concat" in text_df:
        text_series = text_df["concat"]
    elif "text" in text_df:
        text_series = text_df["text"]
    else:
        print("Invalid `text_df`.")
        return
    text_series = text_series.fillna("")
    text_features = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(text_series), batch_size):
            token_output = tokenizer(
                text_series[i : i + batch_size].to_list(),
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            output = text_extractor(**token_output)
            text_features.append(output.last_hidden_state[:, 0].to("cpu"))
            del output
            del token_output
    tensors = {
        "item_id": torch.tensor(text_df["Unnamed: 0"].to_numpy()),
        "text_features": torch.cat(text_features),
    }
    save_file(tensors, csv_path.parent / "text.safetensors")


def preprocess_iamge(csv_path, image_path, device="cpu"):
    image_extractor = convnext_tiny(weights="DEFAULT").to(device).eval()

    image_dict = {}
    to_tensor = ToTensor()
    image_path = Path(image_path)
    if image_path.exists() and image_path.is_dir():
        for image in image_path.glob("*.jpg"):
            image_dict[image.stem] = to_tensor(Image.open(image).convert("RGB"))

    csv_path = Path(csv_path)
    image_df = pd.read_csv(csv_path)

    image_features = []
    with torch.no_grad():
        for asin in image_df["asin"]:
            if asin in image_dict:
                output = image_extractor(
                    image_dict[asin].reshape(1, 3, 236, 236).to(device)
                )
            else:
                output = torch.zeros(1, 1000)
            image_features.append(output.to("cpu"))
            del output

    image_tensors = {
        "item_id": torch.tensor(image_df["Unnamed: 0"].to_numpy()),
        "image_features": torch.cat(image_features),
    }
    save_file(image_tensors, csv_path.parent / "image.safetensors")


def preprocess_data(dataset, types, device="cuda:2"):
    image_path = f"{dataset}/images"
    for _type in types:
        csv_path = f"{dataset}/{_type}.csv"
        if _type == "image":
            preprocess_iamge(csv_path, image_path, device)
        elif _type == "text":
            preprocess_text(csv_path, device)

def load_multimodal_features(dataset_dir):
    print("loading texts...")
    text_features = load_file(f"{dataset_dir}/text.safetensors")["text_features"]
    print("loading images... This will take a few minutes...")
    image_features = load_file(f"{dataset_dir}/image.safetensors")["image_features"]
    return text_features, image_features


def load_multimodal_dataset(
    dataset_dir,
    repetitive,
):
    text_features, image_features = load_multimodal_features(dataset_dir)

    train_data = MultimodalSessionFeatureDataset(
        dataset_dir,
        repetitive,
        "train",
        text_feature_tensor=text_features,
        image_feature_tensor=image_features,
    )
    val_data = MultimodalSessionFeatureDataset(
        dataset_dir,
        repetitive,
        "valid",
        text_feature_tensor=text_features,
        image_feature_tensor=image_features,
    )
    test_data = MultimodalSessionFeatureDataset(
        dataset_dir,
        repetitive,
        "test",
        text_feature_tensor=text_features,
        image_feature_tensor=image_features,
    )

    num_items = max(train_data.max_iid, val_data.max_iid, test_data.max_iid) + 1

    return (train_data, val_data, test_data, num_items)


def preprocess(data_df, repetitive, train=False):
    MAX_LENGTH = 50
    data = []
    prev_sid = -1

    for row in data_df.itertuples():
        if prev_sid != row.sessionId:
            data.append([])
        data[-1].append(row.itemId + 1)
        prev_sid = row.sessionId

    data_item = []
    data_length = []
    data_target = []

    for session in data:
        if train:
            if len(session) <= 3:
                continue
            # if len(session) > MAX_LENGTH:
            #     sub_session = session[:MAX_LENGTH]
            #     data_item.append(sub_session)
            #     data_length.append(float(len(sub_session)))
            #     data_target.append(session[1 : MAX_LENGTH + 1])
            if len(session) > MAX_LENGTH:
                sub_session = session[-MAX_LENGTH:-1]
                data_item.append(sub_session)
                data_length.append(float(len(sub_session)))
                data_target.append(session[-MAX_LENGTH + 1: ])
            else:
                sub_session = session[:-1]
                data_item.append(sub_session)
                data_length.append(float(len(sub_session)))
                data_target.append(session[1:])
        else:
            # for ind in range(3, len(session)):
            #     if not repetitive and session[ind] in session[:ind]:
            #         continue
            #     if ind > MAX_LENGTH:
            #         break
            #     sub_session = session[:ind]
            #     data_item.append(sub_session)
            #     data_length.append(float(len(sub_session)))
            #     data_target.append(session[ind])

            min_length = 3
            if len(session) > min_length:
                # if len(session) > MAX_LENGTH: ind = MAX_LENGTH # 오래된 순서로 50개
                # else: ind = -1
                # if not repetitive and session[ind] in session[:ind]:
                #     continue
                # if ind > MAX_LENGTH:
                #     break
                # sub_session = session[:ind]
                # data_item.append(sub_session)
                # data_length.append(float(len(sub_session)))
                # data_target.append(session[ind])

                if len(session) > MAX_LENGTH: ind = -MAX_LENGTH # 최근 50개
                else: ind = 0
                if not repetitive and session[-1] in session[ind:-1]:
                    continue

                sub_session = session[ind:-1]
                data_item.append(sub_session)
                data_length.append(float(len(sub_session)))
                data_target.append(session[-1])


    return data_item, data_length, data_target


class MultimodalSessionFeatureDataset(Dataset):
    def __init__(
        self, dataset, repetitive, type_, text_feature_tensor, image_feature_tensor
    ) -> None:
        super().__init__()
        assert type_.lower() in [
            "train",
            "valid",
            "test",
        ], '`type_` should be one of: ["train", "valid", "test"]'
        df = pd.read_csv(f"{dataset}/{type_}.csv").fillna("NA")
        train = type_.lower() == "train"

        self.data_iid, self.data_length, self.data_label = preprocess(
            df, repetitive, train=train
        )
        self.max_iid = df["itemId"].max()

        self.image_features = image_feature_tensor
        self.text_features = text_feature_tensor

    def __len__(self):
        return len(self.data_iid)

    def __getitem__(self, index):
        iids = self.data_iid[index]
        # iid = item_id + 1
        if isinstance(index, int):  # iids is a list
            text = self.text_features[[iid - 1 for iid in iids]]
        else:  # iids is a list of list
            text = [
                self.text_features[[iid - 1 for iid in indices]] for indices in iids
            ]
        if isinstance(index, int):
            image = self.image_features[[iid - 1 for iid in iids]]
        else:
            image = [
                self.image_features[[iid - 1 for iid in indices]] for indices in iids
            ]
        return (
            iids,
            self.data_length[index],
            self.data_label[index],
            text,
            image,
        )
    
class InferenceDataset(Dataset):
    pass

class MultimodalInferenceDataset(InferenceDataset):
    pass

if __name__ == "__main__":
    from argparse import ArgumentParser

    from preprocessing import preprocess_and_save
    parser = ArgumentParser()
    parser.add_argument("root",nargs='?', default="amazon_25000", help="Root of dataset folder. Must contain `images/`, `image.csv`, and `text.csv`")
    parser.add_argument("--split_csv_file", help="CSV file to be split into `train.csv`, `valid.csv` and `test.csv`.")
    parser.add_argument("--no_text",action="store_true", help="When set, the text dataset is not processed.")
    parser.add_argument("--no_image",action="store_true", help="When set, the image dataset is not processed")
    args = parser.parse_args()
    types = []
    if not args.no_text:
        types.append("text")
    if not args.no_image:
        types.append("image")
    if args.split_csv_file is not None:
        preprocess_and_save(args.split_csv_file, args.root)
    preprocess_data(args.root, types)