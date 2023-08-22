import torch
import numpy as np


def get_ndcg(indices, targets):
    # print(indices.shape) # [256, 10/20/50]
    # print(targets.shape) # [256]
    targets = targets.view(-1, 1).expand_as(indices) # 512 x items
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1

    dcg = 1 / np.log2(1 + ranks.detach().cpu().numpy())
    idcg = 1 / np.log2(1 + 1)
    ndcg = dcg / idcg

    batch_ndcg = np.sum(ndcg)
    return batch_ndcg


def get_recall(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    
    n_hits = len(hits)
    recall = float(n_hits)
    return recall


def get_mrr(indices, targets):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data
    return mrr.item()


def evaluate(dist, target, K):
    recalls = []
    mrrs = []
    ndcgs = []

    for k_ in K:
        _, indices = torch.topk(dist, k_, dim=-1, largest=False)

        recall = get_recall(indices, target)
        mrr = get_mrr(indices, target)
        ndcg = get_ndcg(indices, target)

        recalls.append(recall)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

    return recalls, mrrs, ndcgs
