import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import convnext_tiny
from transformers import AutoTokenizer, DistilBertForSequenceClassification


from model_util import *


class ProxySR(nn.Module):
    def __init__(
        self, num_items:int, embedding_dim:int, k:int, max_position:int, dropout:float, margin:float, device
    ):
        super(ProxySR, self).__init__()
        self.device = device

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.k = k
        self.max_position = max_position
        self.margin = margin

        # Embeddings
        self.P_emb = nn.Embedding(self.k, self.embedding_dim, max_norm=1.0)
        self.I_emb = nn.Embedding(
            self.num_items + 1, self.embedding_dim, padding_idx=0, max_norm=1.0
        )  # 0 ~ 17089
        self.E_P_emb = nn.Embedding(
            self.max_position, self.embedding_dim, max_norm=1.0
        )  # positional embedding for proxy
        self.E_S_emb = nn.Embedding(
            self.max_position + 1, self.embedding_dim, padding_idx=0, max_norm=1.0
        )  # positional embedding for session

        # Unit normal vector set
        self.V = nn.Embedding(self.k, self.embedding_dim, max_norm=1.0)

        # Point-wise Feed Forward Network
        in_shape = self.embedding_dim
        out_shape = self.k
        self.WP_1 = nn.Linear(in_shape, (in_shape + out_shape) // 2, bias=False)
        self.WP_2 = nn.Linear((in_shape + out_shape) // 2, out_shape, bias=False)

        # Self-attention Network
        self.SAN = SelfAttentionNetwork(
            self.embedding_dim,
            self.embedding_dim,
            self.embedding_dim,
            self.embedding_dim,
            dropout=dropout,
        )

        self.distFunction = squaredEuclideanDistance

        self.dropout = nn.Dropout(dropout)

        # Initialization
        emb_std = 0.01
        nn.init.normal_(self.P_emb.weight, 0.0, emb_std)
        nn.init.normal_(self.I_emb.weight, 0.0, emb_std)
        nn.init.normal_(self.E_P_emb.weight, 0.0, emb_std)
        nn.init.normal_(self.E_S_emb.weight, 0.0, emb_std)
        nn.init.normal_(self.V.weight, 0.0, emb_std)

        W_weight_std = 1.0
        nn.init.normal_(self.WP_1.weight.data, 0.0, W_weight_std)
        nn.init.normal_(self.WP_2.weight.data, 0.0, W_weight_std)

        # make the norm = 1
        self.I_emb.weight.data.div_(
            torch.max(
                torch.norm(self.I_emb.weight.data, 2, 1, True), torch.FloatTensor([1.0])
            ).expand_as(self.I_emb.weight.data)
        )
        self.P_emb.weight.data.div_(
            torch.max(
                torch.norm(self.P_emb.weight.data, 2, 1, True), torch.FloatTensor([1.0])
            ).expand_as(self.P_emb.weight.data)
        )
        self.E_P_emb.weight.data.div_(
            torch.max(
                torch.norm(self.E_P_emb.weight.data, 2, 1, True),
                torch.FloatTensor([1.0]),
            ).expand_as(self.E_P_emb.weight.data)
        )
        self.E_S_emb.weight.data.div_(
            torch.max(
                torch.norm(self.E_S_emb.weight.data, 2, 1, True),
                torch.FloatTensor([1.0]),
            ).expand_as(self.E_S_emb.weight.data)
        )
        self.V.weight.data = F.normalize(self.V.weight.data, p=2, dim=-1)

    def ProxySelection(self, isValidPos, session_embs, E_P_emb, lengths, tau, P_emb, V):
        ## ProxySelction
        # fP: Session encoder
        X = self.dropout(isValidPos.unsqueeze(-1) * (session_embs + E_P_emb))
        X = self.WP_2(torch.nn.LeakyReLU(0.1)(self.WP_1(X)))
        fP = torch.sum(X, dim=1) / lengths.view(-1, 1)
        alpha = fP
        pi = torch.softmax(alpha / tau, dim=-1)

        # select p_s, v from the sets
        p_s = summarize_vectors(P_emb, pi)  # b * s * n
        v = F.normalize(torch.matmul(pi, V), p=2, dim=-1)  # b * s * n

        return p_s, v

    def SIE(self, session_embs, E_S_emb, max_length, batch_size, session):
        ### Short-term Interest Encoder
        X = session_embs + E_S_emb

        # mask for auto-regressive property
        subsequent_mask = torch.triu(
            torch.ones((max_length, max_length), device=self.device, dtype=torch.uint8),
            diagonal=1,
        )
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # batch_size * max_length * max_length
        padding_mask = session.eq(0)  # batch_size * max_length
        padding_mask = padding_mask.unsqueeze(1).expand(
            -1, max_length, -1
        )  # batch_size * max_length * max_length
        mask = (subsequent_mask + padding_mask).gt(0)
        mask[:, :, 0] = 0
        non_pad_mask = session.ne(0).type(torch.float).unsqueeze(-1)

        # f_S: Session encoder
        SAN_input = X
        fS = self.SAN(SAN_input, non_pad_mask=non_pad_mask, slf_attn_mask=mask)

        return fS

    def forward(self, session, lengths, tau, isValidSession, train):
        # Unit vector constraint
        self.V.weight.data = F.normalize(self.V.weight.data, p=2, dim=-1)

        batch_size = session.size()[0]
        max_length = session.size()[1]
        lengths = lengths.to(torch.float)

        isValidPos = (session != 0.0).to(torch.float)  # 0은 padding을 의미함.

        # Embeddings
        I_emb = self.I_emb(
            (torch.arange(self.num_items + 1)).to(self.device)
        )  # num_items * dim
        P_emb = self.P_emb(torch.arange(self.k, device=self.device))  # k * dim
        session_embs = self.I_emb(session)  # batch_size * max_length * dim
        E_P_emb = self.E_P_emb(
            torch.arange(session.shape[1]).to(self.device)
            + (self.max_position - session.shape[1])
        )  # like tensor([40, 41, .. , 49])
        E_S_emb = self.E_S_emb(
            torch.relu(
                torch.arange(session.shape[1]).to(self.device)
                + 1
                - (max_length - lengths.to(torch.long).view(-1, 1))
            )
        )  # [0 0 0 0 1 2 3 4 5 6]

        # Unit normal vectors for the proxy
        V = self.V(torch.arange(self.k, device=self.device))  # k * dim

        ## Proxy Selection
        p_s, v = self.ProxySelection(
            isValidPos, session_embs, E_P_emb, lengths, tau, P_emb, V
        )

        ## Short-term Interest Encoder (SIE)
        s_s = self.SIE(
            session_embs, E_S_emb, max_length, batch_size, session
        )  # b * max_dist(각 item이라고 생각할 수 있겠네) * dim

        # Projection of the short-term interest on the proxy's space
        s_s_proj = s_s - v.unsqueeze(1) * torch.sum(
            v.unsqueeze(1) * s_s, dim=-1, keepdim=True
        )

        if train:
            return self.train_phase(
                batch_size, isValidSession, p_s, s_s_proj, I_emb, v, max_length
            )
        else:
            return self.test_phase(
                batch_size, isValidSession, p_s, s_s_proj[:, -1, :], I_emb, v
            )

    def train_phase(
        self, batch_size, isValidSession, p_s, s_s_proj, I_emb, v, max_length
    ):
        # s_s_proj: batch * max_length * dim (session내 모든 아이템에 대한 embedding)

        distance = []
        orthogonal = torch.abs(torch.sum(v * p_s, dim=-1)) / torch.norm(p_s, dim=-1)
        for i in range(batch_size):
            I_emb_proj = I_emb - v[i, :].unsqueeze(0) * torch.sum(
                v[i, :].unsqueeze(0) * I_emb, dim=-1, keepdim=True
            )
            s_s_t_proj = s_s_proj[i, :, :]

            for t in range(max_length):
                if not isValidSession[i * max_length + t]:
                    continue
                final_sess = (
                    p_s[i, :] + s_s_t_proj[t, :]
                )  # session 내 t번째 item으로 이해 가능 (128 dim) -> 이걸로 t+1을 맞추면 되는구나.
                # print('final_sess.unsqueeze(0).shape', final_sess.unsqueeze(0).shape)
                # print('item_emb.shape', I_emb_proj.shape)
                dist = self.distFunction(final_sess.unsqueeze(0), I_emb_proj)
                distance.append(dist)

        distance = torch.stack(distance)

        return distance, orthogonal

    def test_phase(self, batch_size, isValidSession, p_s, s_s_proj, I_emb, v):
        # s_s_proj: batch x dim (session내 마지막 item으로부터 예측함)

        distance = []
        final_sess = p_s + s_s_proj  # batch x dim like (512,128)

        for i in range(batch_size):
            if not isValidSession[i]:
                continue
            I_emb_proj = I_emb - v[i, :].unsqueeze(0) * torch.sum(
                v[i, :].unsqueeze(0) * I_emb, dim=-1, keepdim=True
            )
            dist = self.distFunction(final_sess[i].unsqueeze(0), I_emb_proj)
            distance.append(dist)
        distance = torch.stack(distance)

        return distance
    
    def metadata(self):
        return {
        "device":str(self.device),
        "num_items": str(self.num_items), 
        "embedding_dim":str(self.embedding_dim), 
        "k":str(self.k),
        "max_position":str(self.max_position),
        "dropout":str(self.dropout.p),
        "margin":str(self.margin),
        }


class MultimodalProxySR(ProxySR):
    def __init__(
        self,
        num_items:int,
        embedding_dim:int,
        k:int,
        max_position:int,
        dropout:float,
        margin:float,
        device,
        *,
        image_weight:float=1.0,
        text_weight:float=1.0,
        image_dim:int=1000,
        text_dim:int=768
    ):
        super().__init__(
            num_items, embedding_dim, k, max_position, dropout, margin, device
        )
        self.image_emb = nn.Linear(image_dim, embedding_dim)  # based on convnext_tiny
        self.text_emb = nn.Linear(
            text_dim, embedding_dim
        )  # based on distilbert-base-uncased
        self.image_weight = image_weight
        self.text_weight = text_weight

    def forward(
        self,
        session,
        text_features,
        image_features,
        lengths,
        tau,
        isValidSession,
        train,
    ):
        # Unit vector constraint
        self.V.weight.data = F.normalize(self.V.weight.data, p=2, dim=-1)

        batch_size = session.size()[0]
        max_length = session.size()[1]
        lengths = lengths.to(torch.float)

        isValidPos = (session != 0.0).to(
            torch.float
        )  # 0은 padding을 의미함. -> Data에서도 그런 게 맞나?
        # image_features = self.image_extractor(images.reshape(-1, 3, 236, 236)).reshape(
        #     (batch_size, max_length, self.embedding_dim)
        # )

        # Embeddings
        I_emb = self.I_emb(
            (torch.arange(self.num_items + 1)).to(self.device)
        )  # num_items * dim
        P_emb = self.P_emb(torch.arange(self.k, device=self.device))  # k * dim
        session_embs = (
            self.I_emb(session)
            + self.image_weight * self.image_emb(image_features)
            + self.text_weight * self.text_emb(text_features)
        )  # batch_size * max_length * dim

        E_P_emb = self.E_P_emb(
            torch.arange(session.shape[1]).to(self.device)
            + (self.max_position - session.shape[1])
        )  # like tensor([40, 41, .. , 49])
        E_S_emb = self.E_S_emb(
            torch.relu(
                torch.arange(session.shape[1]).to(self.device)
                + 1
                - (max_length - lengths.to(torch.long).view(-1, 1))
            )
        )  # [0 0 0 0 1 2 3 4 5 6]

        # Unit normal vectors for the proxy
        V = self.V(torch.arange(self.k, device=self.device))  # k * dim

        ## Proxy Selection
        p_s, v = self.ProxySelection(
            isValidPos, session_embs, E_P_emb, lengths, tau, P_emb, V
        )

        ## Short-term Interest Encoder (SIE)
        s_s = self.SIE(
            session_embs, E_S_emb, max_length, batch_size, session
        )  # b * max_dist(각 item이라고 생각할 수 있겠네) * dim

        # Projection of the short-term interest on the proxy's space
        s_s_proj = s_s - v.unsqueeze(1) * torch.sum(
            v.unsqueeze(1) * s_s, dim=-1, keepdim=True
        )

        if train:
            return self.train_phase(
                batch_size, isValidSession, p_s, s_s_proj, I_emb, v, max_length
            )
        else:
            return self.test_phase(
                batch_size, isValidSession, p_s, s_s_proj[:, -1, :], I_emb, v
            )

    def get_tokenizer(self):
        return self.tokenizer

    def metadata(self):
        metadata = super().metadata()
        metadata.update({
            "image_weight": str(self.image_weight),
            "text_weight" : str(self.text_weight),
            "image_dim":str(self.image_emb.in_features),
            "text_dim":str(self.text_emb.in_features)
        })
        return metadata