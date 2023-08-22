import pandas as pd
from torch.utils.data import Dataset


from preprocessed_dataset import preprocess

MAX_LENGTH = 50


class SessionDataset(Dataset):
    def __init__(self, dataset, repetitive, type_) -> None:
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

    def __len__(self):
        return len(self.data_iid)

    def __getitem__(self, index):
        return self.data_iid[index], self.data_length[index], self.data_label[index]


def load_dataset(dataset, repetitive):
    train_data = SessionDataset(dataset, repetitive, "train")
    val_data = SessionDataset(dataset, repetitive, "valid")
    test_data = SessionDataset(dataset, repetitive, "test")

    num_items = max(train_data.max_iid, val_data.max_iid, test_data.max_iid) + 1

    return (train_data, val_data, test_data, num_items)


# def load_multimodal_dataset(
#     dataset,
#     repetitive,
# ):
#     print("loading texts...")
#     text_df = pd.read_csv("amazon_25000/text25000.csv").set_index("asin")
#     concat_text_df = text_df.apply(
#         lambda row: row.title + ("" if pd.isna(row.description) else row.description),
#         axis=1,
#     ).to_frame("text")

#     print("loading images... This will take a few minutes...")
#     image_dict = {}
#     to_tensor = ToTensor()
#     dataset_path = Path(f"{dataset}")
#     image_path = dataset_path / "images"
#     if image_path.exists() and image_path.is_dir():
#         image_cache = (dataset_path/"image_dict.pth") 
#         if image_cache.exists():
#             image_dict = torch.load(image_cache)
#         else:
#             for image in image_path.glob("*.jpg"):
#                 image_dict[image.stem] = to_tensor(Image.open(image).convert('RGB'))

#     train_data = MultimodalSessionDataset(
#         dataset, repetitive, "train", text_df=concat_text_df, image_dict=image_dict
#     )
#     val_data = MultimodalSessionDataset(
#         dataset, repetitive, "valid", text_df=concat_text_df, image_dict=image_dict
#     )
#     test_data = MultimodalSessionDataset(
#         dataset, repetitive, "test", text_df=concat_text_df, image_dict=image_dict
#     )
#     # torch.save(train_data, image_path / "train.pkl")
#     # torch.save(val_data, image_path / "valid.pkl")
#     # torch.save(test_data, image_path / "test.pkl")

#     num_items = max(train_data.max_iid, val_data.max_iid, test_data.max_iid) + 1

#     return (train_data, val_data, test_data, num_items)


# def preprocess(data_df, repetitive, train=False, *, return_text=False, image_dict=None):
#     use_text = "text" in data_df and return_text
#     use_image = image_dict is not None
#     is_multimodal = use_text or use_image

#     data = []
#     data_index = []
#     data_asin = []

#     prev_sid = -1
#     # sid = data_df['sessionId']
#     # iid = data_df['itemId']
#     # for s, i in zip(sid, iid):
#     #     if prev_sid != s:
#     #         data.append([])
#     #     data[-1].append(i + 1)
#     #     prev_sid = s

#     for index, row in enumerate(data_df.itertuples()):
#         if prev_sid != row.sessionId:
#             if is_multimodal:
#                 data_index.append([])
#                 data_asin.append([])
#             data.append([])
#         if is_multimodal:
#             data_index[-1].append(row.text)#index)
#             data_asin[-1].append(row.asin)
#         data[-1].append(row.itemId + 1)
#         prev_sid = row.sessionId

#     data_item = []
#     data_length = []
#     data_target = []

#     data_text = []
#     data_image = []

#     for session_index, session in enumerate(data):
#         if train:
#             if len(session) <= 3:
#                 continue
#             if len(session) > MAX_LENGTH:
#                 sub_session = session[:MAX_LENGTH]
#                 data_item.append(sub_session)
#                 data_length.append(float(len(sub_session)))
#                 data_target.append(session[1 : MAX_LENGTH + 1])
#                 if use_text:
#                     # data_indices = data_index[session_index][:MAX_LENGTH]
#                     # data_text.append(data_df["text"][data_indices].to_list())
#                     data_text.append(data_index[session_index][:MAX_LENGTH])
#                 if use_image:
#                     image_asins = data_asin[session_index][:MAX_LENGTH]
#                     data_image.append(
#                         [
#                             image_dict.get(name, torch.zeros(3, 236, 236))
#                             for name in image_asins
#                         ]
#                     )
#             else:
#                 sub_session = session[:-1]
#                 data_item.append(sub_session)
#                 data_length.append(float(len(sub_session)))
#                 data_target.append(session[1:])
#                 if use_text:
#                     # data_indices = data_index[session_index][:-1]
#                     # data_text.append(data_df["text"][data_indices].to_list())
#                     data_text.append(data_index[session_index][:-1])
#                 if use_image:
#                     image_asins = data_asin[session_index][:-1]
#                     data_image.append(
#                         [
#                             image_dict.get(name, torch.zeros(3, 236, 236))
#                             for name in image_asins
#                         ]
#                     )

#         else:
#             for ind in range(3, len(session)):
#                 if not repetitive and session[ind] in session[:ind]:
#                     continue
#                 if ind > MAX_LENGTH:
#                     break
#                 sub_session = session[:ind]
#                 data_item.append(sub_session)
#                 data_length.append(float(len(sub_session)))
#                 data_target.append(session[ind])
#                 if use_text:
#                     # data_indices = data_index[session_index][:ind]
#                     # data_text.append(data_df["text"][data_indices].to_list())
#                     data_text.append(data_index[session_index][:ind])

#                 if use_image:
#                     image_asins = data_asin[session_index][:ind]
#                     data_image.append(
#                         [
#                             image_dict.get(name, torch.zeros(3, 236, 236))
#                             for name in image_asins
#                         ]
#                     )

#     optional = {}
#     if is_multimodal:
#         if use_image:
#             optional["image"] = data_image
#         if use_text:
#             optional["text"] = data_text
#         return data_item, data_length, data_target, optional
#     return data_item, data_length, data_target


# class MultimodalSessionDataset(Dataset):
#     def __init__(self, dataset, repetitive, type_, text_df, image_dict) -> None:
#         super().__init__()
#         assert type_.lower() in [
#             "train",
#             "valid",
#             "test",
#         ], '`type_` should be one of: ["train", "valid", "test"]'
#         df = pd.read_csv(f"{dataset}/{type_}.csv").fillna("NA")
#         train = type_.lower() == "train"

#         joined_text_df = df.join(text_df, on="asin")[
#             ["reviewerID", "asin", "itemId", "sessionId", "text"]
#         ]
#         self.data_iid, self.data_length, self.data_label = preprocess(
#             joined_text_df,
#             repetitive,
#             train=train,
#             return_text=False,
#             image_dict=None,
#         )
#         self.max_iid = df["itemId"].max()

#         # if "text" in optional:
#         #     self.text = optional["text"]
#         # else:
#         #     raise AssertionError("`preprocess` should return optional text.")

#         # if "image" in optional:
#         #     self.image = optional["image"]
#         # else:
#         #     raise AssertionError("`preprocess` should return optional image.")
#         self.df = joined_text_df[["asin", "text", "itemId"]].drop_duplicates(subset="itemId").set_index("itemId")
#         self.df["text"]=self.df["text"].fillna("")
#         self.image_dict = image_dict

#     def __len__(self):
#         return len(self.data_iid)

#     def __getitem__(self, index):
#         iids = self.data_iid[index]
#         # iid = item_id + 1
#         if isinstance(index, int):
#             text = [self.df["text"][iid-1] for iid in iids]
#         else:
#             text = [self.df["text"][[iid -1 for iid in indices]].to_list() for indices in iids]
#         if isinstance(index, int):
#             image = [self.image_dict.get(self.df["asin"][iid-1], torch.zeros(3,236,236)) for iid in iids]
#         else:
#             image = [[self.image_dict.get(self.df["asin"][iid-1], torch.zeros(3,236,236)) for iid in item_ids] for item_ids in iids]
#         return (
#             iids,
#             self.data_length[index],
#             self.data_label[index],
#             text,
#             image,
#         )
