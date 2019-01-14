import torch as t
from.utils import *




class EncoderLayer(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_head, dropout)
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, input_embedding, self_attention_mask, pad_mask):
        net = self.self_attention(input_embedding, self_attention_mask, pad_mask)
        net = self.feed_forward(net)
        return net


class DecoderLayer(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_head, dropout)
        self.dot_attention = DotAttention(hidden_size, num_head, dropout)
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, query_embedding, key_embedding, value_embedding, self_attention_mask, dot_attention_mask, pad_mask):
        net = self.self_attention(query_embedding, self_attention_mask, pad_mask)
        net = self.dot_attention(net, key_embedding, value_embedding, dot_attention_mask, pad_mask)
        net = self.feed_forward(net)
        return net


class Embedding(t.nn.Module):
    def __init__(self, embedding_num, embedding_size, max_lenth, max_batch_size, dropout, padding_idx=0):
        super(Embedding, self).__init__()
        self.word_embedding = t.nn.Embedding(embedding_num, embedding_size, padding_idx)
        self.position_embedding = t.nn.Embedding(max_lenth, embedding_size)
        self.layer_norm = t.nn.LayerNorm(embedding_size)
        self.dropout = t.nn.Dropout(dropout)
        self.init_position_feature(max_lenth, max_batch_size)

    def init_position_feature(self, max_lenth, max_batch_size):
        ## register postion_feature
        position_feature = t.arange(max_lenth, dtype=t.long)
        position_feature.unsqueeze_(0)
        position_feature = position_feature.expand((max_batch_size, max_lenth))
        self.register_buffer('position_feature', position_feature)

    def forward(self, input_token, pad_mask):
        ## get shape and device
        batch_size, seq_lenth = input_token.size()
        device = input_token.device
        ## inference
        word_embedding = self.word_embedding(input_token)
        position_embedding = self.position_embedding(self.position_feature[:batch_size, :seq_lenth].to(device))
        embedding = word_embedding + position_embedding
        embedding *= pad_mask.unsqueeze(-1)
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class BertEmbedding(t.nn.Module):
    def __init__(self, embedding_num, embedding_size, max_lenth, max_batch_size, dropout, padding_idx=0):
        super(BertEmbedding, self).__init__()
        self.word_embedding = t.nn.Embedding(embedding_num, embedding_size, padding_idx)
        self.position_embedding = t.nn.Embedding(max_lenth, embedding_size)
        self.segment_embedding = t.nn.Embedding(3, embedding_size, padding_idx)
        self.layer_norm = t.nn.LayerNorm(embedding_size)
        self.dropout = t.nn.Dropout(dropout)
        self.init_position_feature(max_lenth, max_batch_size)

    def init_position_feature(self, max_lenth, max_batch_size):
        ## register postion_feature
        position_feature = t.arange(max_lenth, dtype=t.long)
        position_feature.unsqueeze_(0)
        position_feature = position_feature.expand((max_batch_size, max_lenth))
        self.register_buffer('position_feature', position_feature)

    def forward(self, input_token, segment_label, pad_mask):
        ## get shape and device
        batch_size, seq_lenth = input_token.size()
        device = input_token.device
        ## inference
        word_embedding = self.word_embedding(input_token)
        position_embedding = self.position_embedding(self.position_feature[:batch_size, :seq_lenth].to(device))
        segment_embedding = self.segment_embedding(segment_label)
        embedding = word_embedding + position_embedding + segment_embedding
        embedding *= pad_mask.unsqueeze(-1)
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding

