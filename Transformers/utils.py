import torch as t
import math
import numpy as np


class FeedForward(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear = t.nn.Sequential(
            t.nn.Linear(hidden_size, hidden_size*3),
            GeLU(),
            t.nn.Linear(hidden_size*3, hidden_size)
        )
        self.layer_norm = t.nn.LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, embedding):
        net = self.linear(embedding)
        net = self.layer_norm(net + embedding)
        net = self.dropout(net)
        return net


class GeLU(t.nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + t.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * t.pow(x, 3))))


def get_pad_mask(input):
    return input.data.ne(0).float()


def get_self_attention_mask(input):
    lenth = input.size(-1)
    mask = get_pad_mask(input)
    mask.unsqueeze_(1)
    mask = mask.expand(-1, lenth, -1)
    return mask

def get_dot_attention_mask(key, query):
    lenth = query.size(-1)
    mask = get_pad_mask(key)
    mask.unsqueeze_(1)
    mask = mask.expand(-1, lenth, -1)
    return mask

def get_sequential_mask(query, key):
    device = query.device
    batch_size, query_lenth = query.size()
    batch_size, key_lenth = key.size()
    sequential_mask = t.tril(t.ones(query_lenth, query_lenth, device=device))
    sequential_mask.unsqueeze_(0)
    sequential_mask = sequential_mask.expand(batch_size, -1, -1)
    return sequential_mask



class OriDotAttention(t.nn.Module):
    def __init__(self, dropout, hidden_size):
        super(OriDotAttention, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.C = np.sqrt(hidden_size)

    def forward(self, query, key, value, attention_mask, pad_mask):
        """

        :param query: b, lq, h
        :param key: b, lk, h
        :param value: b, lv, h
        :param attention_mask: b, lq, lk
        :param pad_mask: b, lq
        :return: b, lq, h
        """
        score = t.matmul(query, key.transpose(-1, -2)) / self.C
        score.masked_fill_(attention_mask == 0, -float('inf'))
        score = t.nn.functional.softmax(score, -1)
        score = self.dropout(score)
        weighted = t.bmm(score, value)
        weighted *= pad_mask.unsqueeze(-1)
        return weighted


class MultiHeadSelfAttention(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.head_size = int(self.hidden_size / self.num_head)
        self.all_head_size = self.head_size * self.num_head

        self.key_projection = t.nn.Linear(self.hidden_size, self.all_head_size)
        self.query_projection = t.nn.Linear(self.hidden_size, self.all_head_size)
        self.value_projection = t.nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = t.nn.Dropout(dropout)
        self.C = np.sqrt(self.head_size)

    def split_head(self, x):
        shape = x.size()[:-1] + (self.num_head, self.head_size)
        return x.view(*shape).permute(0, 2, 1, 3)

    def forward(self, embedding, self_attention_mask, pad_mask):
        """

        :param embedding: b, l, h
        :param self_attention_mask: b, l, l
        :param pad_mask: b, l
        :return:
        """
        query = self.query_projection(embedding)
        key = self.key_projection(embedding)
        value = self.value_projection(embedding)

        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        attention_score = t.matmul(query, key.transpose(-1, -2)) / self.C
        attention_score.masked_fill_(self_attention_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1) == 0, -float('inf'))

        attention_score = t.nn.functional.softmax(attention_score, -1)
        attention_score = self.dropout(attention_score)

        weighted = t.matmul(attention_score, value)
        weighted = weighted.permute(0, 2, 1, 3).contiguous()
        shape = weighted.size()[:-2] + (self.all_head_size, )
        output = weighted.view(*shape)
        return output * pad_mask.unsqueeze(-1)


class MultiHeadDotAttention(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(MultiHeadDotAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.head_size = int(self.hidden_size / self.num_head)
        self.all_head_size = self.head_size * self.num_head

        self.key_projection = t.nn.Linear(self.hidden_size, self.all_head_size)
        self.query_projection = t.nn.Linear(self.hidden_size, self.all_head_size)
        self.value_projection = t.nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = t.nn.Dropout(dropout)
        self.C = np.sqrt(self.head_size)

    def split_head(self, x):
        shape = x.size()[:-1] + (self.num_head, self.head_size)
        return x.view(*shape).permute(0, 2, 1, 3)

    def forward(self, query_embedding, key_embedding, value_embedding, dot_attention_mask, pad_mask):
        """

        :param query_embedding:  b, lq, h
        :param key_embedding:  b, lk, h
        :param value_embedding: b, lv, h
        :param dot_attention_mask:  b, lq, lk
        :param pad_mask:  b, lq
        :return:  b, lq, h
        """
        query = self.query_projection(query_embedding)
        key = self.key_projection(key_embedding)
        value = self.value_projection(value_embedding)

        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        attention_score = t.matmul(query, key.transpose(-1, -2)) / self.C
        attention_score.masked_fill_(dot_attention_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1) == 0, -float('inf'))

        attention_score = t.nn.functional.softmax(attention_score, -1)
        attention_score = self.dropout(attention_score)

        weighted = t.matmul(attention_score, value)
        weighted = weighted.permute(0, 2, 1, 3).contiguous()
        shape = weighted.size()[:-2] + (self.all_head_size, )
        output = weighted.view(*shape)
        return output * pad_mask.unsqueeze(-1)


class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(SelfAttention, self).__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_head, dropout)
        self.layer_norm = t.nn.LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, embedding, self_attention_mask, pad_mask):
        net = self.self_attention(embedding, self_attention_mask, pad_mask)
        net = self.layer_norm(net + embedding)
        net = self.dropout(net)
        return net


class DotAttention(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(DotAttention, self).__init__()
        self.dot_attention = MultiHeadDotAttention(hidden_size, num_head, dropout)
        self.layer_norm = t.nn.LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query_embedding, key_embedding, value_embedding, self_attention_mask, pad_mask):
        net = self.dot_attention(query_embedding, key_embedding, value_embedding, self_attention_mask, pad_mask)
        net = self.layer_norm(net + query_embedding)
        net = self.dropout(net)
        return net

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)