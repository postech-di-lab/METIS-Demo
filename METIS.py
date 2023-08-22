from random import shuffle
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from safetensors import safe_open
from safetensors.torch import save_model, load_model

import metric
from model import ProxySR, MultimodalProxySR


class METIS:
    def __init__(self, args, num_items, *, multimodal_option=None, log_file=None, model_file = None, model_save_dir = None):

        self.is_multimodal = multimodal_option is not None
        # model
        if model_file is not None:
            self.model = load_model(self.model, model_file)
        else:
            if self.is_multimodal:
                self.model = MultimodalProxySR(
                    num_items,
                    args.embed_dim,
                    args.k,
                    args.max_position,
                    args.dropout_rate,
                    args.margin,
                    args.device,
                    image_weight=multimodal_option["image_weight"],
                    text_weight=multimodal_option["text_weight"],
                )
            else:
                self.model = ProxySR(
                    num_items,
                    args.embed_dim,
                    args.k,
                    args.max_position,
                    args.dropout_rate,
                    args.margin,
                    args.device,
                )
        self.model = self.model.to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr)

        # hyper-parameters
        self.num_epoch = args.num_epoch
        self.t0 = args.t0
        self.te = args.te
        self.E = args.E
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.K = [10, 20, 50] # [5, 10, 20]
        self.device = args.device
        self.repetitive = args.repetitive
        self.margin = args.margin
        self.lambda_dist = args.lambda_dist
        self.lambda_orthog = args.lambda_orthog
        self.patience = args.patience
        self.tau = 3
        self.log_file = log_file

        self.datetime = datetime.today()
        self.model_save_dir = model_save_dir

    def run(self, train_data, val_data=None, test_data=None):
        mode = "Validation"
        if val_data is None:
            mode = "Test"
            val_data, test_data = test_data, val_data

        best_recall50 = 0
        best_epoch = 0

        for epoch in range(0, self.num_epoch):
            print("", file=self.log_file)
            print("Epoch: " + str(epoch), file=self.log_file)
            # print epoch to screen
            print("Epoch: " + str(epoch))


            # temperature annealing
            self.tau = max(
                self.t0 * ((self.te / self.t0) ** (float(epoch) / self.E)), self.te
            )

            self.train_model(train_data)
            if val_data is not None:
                print("", file=self.log_file)
                print(f"{mode} data:", file=self.log_file)

                recalls, mrrs, ndcgs = self.validate(val_data)

                if recalls[2] > best_recall50:
                    best_recall50 = recalls[2]
                    best_epoch = epoch
                    best_recalls = list(recalls)
                    best_mrrs = list(mrrs)
                    best_ndcg = list(ndcgs)
                    if self.model_save_dir is not None:
                        model_path = self.model_save_dir/f"{'multimodal' if self.is_multimodal else 'singlemodal'}/{self.datetime.isoformat(timespec='seconds')}"
                        if not model_path.exists():
                            model_path.mkdir(parents=True)
                        save_model(self.model, 
                               model_path/f"epoch_{best_epoch}.safetensors",
                               metadata=self.model.metadata())
                    print("", file=self.log_file)
                    print(
                        f"Best Recall@{self.K[0]} : "
                        + str(round(best_recalls[0], 4))
                        + f", Best MRR@{self.K[0]} : "
                        + str(round(best_mrrs[0], 4))
                        + f", Best nDCG@{self.K[0]} : "
                        + str(round(best_ndcg[0], 4)),
                        file=self.log_file,
                    )
                    print(
                        f"Best Recall@{self.K[1]}: "
                        + str(round(best_recalls[1], 4))
                        + f", Best MRR@{self.K[1]}: "
                        + str(round(best_mrrs[1], 4))
                        + f", Best nDCG@{self.K[1]} : "
                        + str(round(best_ndcg[1], 4)),
                        file=self.log_file,
                    )
                    print(
                        f"Best Recall@{self.K[2]}: "
                        + str(round(best_recalls[2], 4))
                        + f", Best MRR@{self.K[2]}: "
                        + str(round(best_mrrs[2], 4))
                        + f", Best nDCG@{self.K[2]} : "
                        + str(round(best_ndcg[2], 4)),
                        file=self.log_file,
                    )

                    # print to screen
                    for k in self.K:
                        print(
                            f"Best Recall@{k} : "
                            + str(round(best_recalls[0], 4))
                            + f"\n Best MRR@{k} : "
                            + str(round(best_mrrs[0], 4))
                            + f"\n Best nDCG@{k} : "
                            + str(round(best_ndcg[0], 4)),
                        )

            if test_data is not None:
                print("", file=self.log_file)
                print("Test data:", file=self.log_file)
                test_recalls, test_mrrs, test_ndcgs = self.validate(test_data)
                print("", file=self.log_file)
                print(
                f"Test Recall@{self.K[0]} : "
                + str(round(test_recalls[0], 4))
                + f", Test MRR@{self.K[0]} : "
                + str(round(test_mrrs[0], 4))
                + f", Test nDCG@{self.K[0]} : "
                + str(round(test_ndcgs[0], 4)),
                file=self.log_file,
            )
                print(
                f"Test Recall@{self.K[1]}: "
                + str(round(test_recalls[1], 4))
                + f", Test MRR@{self.K[1]}: "
                + str(round(test_mrrs[1], 4))
                + f", Test nDCG@{self.K[1]} : "
                + str(round(test_ndcgs[1], 4)),
                file=self.log_file,
            )
                print(
                f"Test Recall@{self.K[2]}: "
                + str(round(test_recalls[2], 4))
                + f", Test MRR@{self.K[2]}: "
                + str(round(test_mrrs[2], 4))
                + f", Test nDCG@{self.K[2]} : "
                + str(round(test_ndcgs[2], 4)),
                file=self.log_file,
            )
            if best_epoch + self.patience < epoch:
                break
        return best_recall50

    def train_model(self, train_data):
        model = self.model
        optimizer = self.optimizer
        batch_size = self.batch_size
        tau = self.tau
        device = self.device
        model.train()
        sum_epoch_target_loss = 0

        batch_index = list(
            range(0, len(train_data), batch_size)
        )  # TODO: 올바른 shuffle 사용할 수 있게 바꾸기
        shuffle(batch_index)
        for i in tqdm(batch_index, leave = False):
            if i + batch_size >= len(train_data):
                train_batch = train_data[i:]
            else:
                train_batch = train_data[i : i + batch_size]

            sess, length, target = train_batch[:3]
            if self.is_multimodal:
                text, image = train_batch[3:]
                text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True).to(
                    device
                )
                image = torch.nn.utils.rnn.pad_sequence(image, batch_first=True).to(
                    device
                )
            max_length = int(max(length))  # -> batch마다 max_length가 다를 수 있음.
            sess = torch.tensor(self.pad(sess, max_length, 0)).to(
                device
            )  # padding이 들어감.
            target = torch.tensor(self.pad(target, max_length, 0)).to(
                device
            )  # padding이 들어감.
            length = torch.tensor(length).to(self.device)
            # print("session.shape", sess.shape)
            # print("target.shape", target.shape)
            # print("length.shape", length.shape)
            # train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=self.collate_batch)
            # for i, train_batch in enumerate(train_loader):
            #     sess, length, target, max_length = train_batch

            #     sess = torch.tensor(sess).to(self.device)
            #     length = torch.tensor(length).to(self.device)
            #     target = torch.tensor(target).to(self.device)

            # print("batch횟수", batch_cnt)
            optimizer.zero_grad()

            target_flatten = target.view(-1)
            sess_cum = sess.repeat(1, max_length).view(-1, max_length) * torch.tril(
                torch.ones((max_length, max_length), device=device), diagonal=0
            ).repeat(sess.shape[0], 1).to(torch.long)

            length3 = (torch.sum(sess_cum != 0, dim=1) >= 3).to(
                device
            )  # batch 내 각각의 data point(session)이 갖는 아이템의 개수가 3이상인가 아닌가?
            if self.repetitive:
                valid = length3
            else:
                unseen_item = (
                    torch.sum((sess_cum - target_flatten.view(-1, 1)).eq(0), dim=1)
                    .eq(0)
                    .to(device)
                )  # target_flatten이 unseen인지 아닌지에 대한 check
                valid = length3 & unseen_item
            valid_ind = valid.nonzero().squeeze()

            valid_target = target_flatten[valid_ind]
            if self.is_multimodal:
                distance, orthogonal = model(
                    sess, text, image, length, tau, valid, train=True
                )
            else:
                distance, orthogonal = model(sess, length, tau, valid, train=True)
            target_distance = distance[range(len(distance)), valid_target]

            margin_distance = torch.max(
                self.margin + target_distance.unsqueeze(1) - distance,
                torch.zeros_like(distance).to(device),
            )

            margin_distance[:, 0] = 0

            margin_distance, _ = torch.topk(margin_distance, k=100, dim=-1)
            target_loss = torch.mean(margin_distance)

            reg_dist = torch.mean(target_distance)
            reg_orthog = torch.mean(orthogonal)

            # Loss
            loss = (
                target_loss
                + self.lambda_dist * reg_dist
                + self.lambda_orthog * reg_orthog
            )

            loss.backward()
            optimizer.step()

            loss_target = target_loss.item()

            sum_epoch_target_loss += loss_target

            if (i / batch_size) % (len(train_data) / batch_size / 5) == (
                len(train_data) / batch_size / 5
            ) - 1:
                print(
                    "Loss: "
                    + str(round(loss_target, 4))
                    + " (avg: "
                    + str(round(sum_epoch_target_loss / (i / batch_size + 1), 4))
                    + ")"
                )

            del loss
            del sess
            del sess_cum
            del length
            del length3
            del valid
            del target
            del train_batch
            if self.is_multimodal:
                del text
                del image

        # Done
        self.model = model
        self.optimizer = optimizer

    def validate(self, val_data):
        model = self.model
        K = self.K
        tau = self.tau
        device = self.device

        # validataion

        model.eval()

        recalls = [0.0, 0.0, 0.0]
        mrrs = [0.0, 0.0, 0.0]
        ndcgs = [0.0, 0.0, 0.0]

        len_val = 0

        # val_loader = DataLoader(val_data,batch_size=self.val_batch_size,shuffle=False)
        with torch.no_grad():
            # for val_batch in val_loader:
            for i in range(0, len(val_data), self.val_batch_size):
                if i + self.val_batch_size >= len(val_data):
                    val_batch = val_data[i:]
                else:
                    val_batch = val_data[i : i + self.val_batch_size]

                sess, length, target = val_batch[:3]  # zip(*val_batch)
                if self.is_multimodal:
                    text, image = val_batch[3:]
                    text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True).to(
                        device
                    )
                    image = torch.nn.utils.rnn.pad_sequence(image, batch_first=True).to(
                        device
                    )
                max_length = int(max(length))
                sess = torch.tensor(self.pad(sess, max_length, 0)).to(device)
                target = torch.tensor(target).to(device)
                length = torch.tensor(length).to(device)

                length3 = (torch.sum(sess != 0, dim=1) >= 3).to(device)
                if self.repetitive:
                    valid = length3
                else:
                    unseen_item = (
                        torch.sum((sess - target.view(-1, 1)).eq(0), dim=1)
                        .eq(0)
                        .to(device)
                    )
                    valid = length3 & unseen_item
                target *= torch.Tensor(valid).to(torch.long).to(device)
                if self.is_multimodal:
                    distance = model(sess, text, image, length, tau, valid, train=False)
                else:
                    distance = model(sess, length, tau, valid, train=False)

                if not self.repetitive:
                    distance[range(len(distance)), sess.t()] = float("inf")
                distance[:, 0] = float("inf")
                len_val += torch.sum(valid).item()
                valid_ind = valid.nonzero().squeeze()
                target = target[valid_ind]

                recall, mrr, ndcg = metric.evaluate(distance, target, K=K)

                recalls[0] += recall[0] # 값더하기
                mrrs[0] += mrr[0]
                ndcgs[0] += ndcg[0]

                recalls[1] += recall[1]
                mrrs[1] += mrr[1]
                ndcgs[1] += ndcg[1]

                recalls[2] += recall[2]
                mrrs[2] += mrr[2]
                ndcgs[2] += ndcg[2]

        recalls = list(map(lambda x: x / len_val, recalls))
        mrrs = list(map(lambda x: x / len_val, mrrs))
        ndcgs = list(map(lambda x: x / len_val, ndcgs))

        # print("")
        # print(
        #     "Recall@{self.K[0]} : "
        #     + str(round(recalls[0], 4))
        #     + ", MRR@{self.K[0]} : "
        #     + str(round(mrrs[0], 4))
        # )
        # print(
        #     "Recall@{self.K[1]}: "
        #     + str(round(recalls[1], 4))
        #     + ", MRR@{self.K[1]}: "
        #     + str(round(mrrs[1], 4))
        # )
        # print(
        #     "Recall@{self.K[2]}: "
        #     + str(round(recalls[2], 4))
        #     + ", MRR@{self.K[2]}: "
        #     + str(round(mrrs[2], 4))
        # )

        return recalls, mrrs, ndcgs

    def predict(self, val_data):
        model = self.model
        tau = self.tau
        device = self.device

        # validataion
        model.eval()
        all_distances = []

        with torch.no_grad():
            for i in range(0, len(val_data), self.val_batch_size):
                if i + self.val_batch_size >= len(val_data):
                    val_batch = val_data[i:]
                else:
                    val_batch = val_data[i : i + self.val_batch_size]

                sess, length, target = val_batch[:3]  # zip(*val_batch)
                if self.is_multimodal:
                    text, image = val_batch[3:]
                    text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True).to(
                        device
                    )
                    image = torch.nn.utils.rnn.pad_sequence(image, batch_first=True).to(
                        device
                    )

                max_length = int(max(length))
                sess = torch.tensor(self.pad(sess, max_length, 0)).to(device)
                target = torch.tensor(target).to(device)
                length = torch.tensor(length).to(device)

                length3 = (torch.sum(sess != 0, dim=1) >= 3).to(device)
                if self.repetitive:
                    valid = length3
                else:
                    unseen_item = (
                        torch.sum((sess - target.view(-1, 1)).eq(0), dim=1)
                        .eq(0)
                        .to(device)
                    )
                    valid = length3 & unseen_item

                if self.is_multimodal:
                    distance = model(sess, text, image, length, tau, valid, train=False)
                else:
                    distance = model(sess, length, tau, valid, train=False)

                if not self.repetitive:
                    distance[range(len(distance)), sess.t()] = float("inf")
                distance[:, 0] = float("inf")

                # save
                all_distances.append(distance)

            # return
            return torch.cat(all_distances)

    def pad(self, l, limit, p):
        max_len = limit
        l = list(
            map(
                lambda x: [p] * (max_len - min(len(x), limit))
                + x[: min(len(x), limit)],
                l,
            )
        )
        return l
    

def load_model_from_file(model_path, device = "cuda" if torch.cuda.is_available() else "cpu"):
    with safe_open(model_path, framework = 'pt') as f:
        metadata = f.metadata()
        if "image_weight" in metadata and "text_weight" in metadata:
            model = MultimodalProxySR(
                int(metadata["num_items"]),
                int(metadata["embedding_dim"]),
                int(metadata["k"]),
                int(metadata["max_position"]),
                0.0, #float(metadata["dropout"]),
                float(metadata["margin"]),
                device,
                image_weight=float(metadata["image_weight"]),
                text_weight=float(metadata["text_weight"]),
                image_dim=int(metadata["image_dim"]),
                text_dim=int(metadata["text_dim"])
            )
        else:
            model = ProxySR(
                int(metadata["num_items"]),
                int(metadata["embedding_dim"]),
                int(metadata["k"]),
                int(metadata["max_position"]),
                0.0,#float(metadata["dropout"]),
                float(metadata["margin"]),
                device,
            )
    load_model(model, model_path)
    return model