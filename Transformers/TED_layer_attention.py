from .layers import *
import ipdb

#TODO
class TransformerEncoder(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout, num_step):
        super(TransformerEncoder, self).__init__()
        self.num_step = num_step
        self.encoder_layer_list = t.nn.ModuleList([EncoderLayer(hidden_size, num_head, dropout) for _ in range(num_step)])

    def forward(self, input_embedding, self_attention_mask, pad_mask):

        for encoder_layer in self.encoder_layer_list:
            input_embedding = encoder_layer(input_embedding, self_attention_mask, pad_mask)

        return input_embedding


class TransformerDecoder(t.nn.Module):
    def __init__(self, hidden_size, num_head, dropout, num_step):
        super(TransformerDecoder, self).__init__()
        self.num_step = num_step
        self.decoder_layer_list = t.nn.ModuleList([DecoderLayer(hidden_size, num_head, dropout) for _ in range(num_step)])

    def forward(self, query_embedding, key_embedding, value_embedding, self_attention_mask, dot_attention_mask, pad_mask):

        for decoder_layer in self.decoder_layer_list:
            query_embedding = decoder_layer(query_embedding, key_embedding, value_embedding, self_attention_mask, dot_attention_mask, pad_mask)

        return query_embedding


class TED(t.nn.Module):
    """
    original transformer,
    """
    def __init__(self, embedding_num, embedding_size, max_lenth, max_batch_size, dropout, hidden_size, num_head, num_step, share_weight=True):
        super(TED, self).__init__()
        self.embedding = Embedding(embedding_num, embedding_size, max_lenth, max_batch_size, dropout)
        self.encoder = TransformerEncoder(hidden_size, num_head, dropout, num_step)
        self.decoder = TransformerDecoder(hidden_size, num_head, dropout, num_step)
        self.projection = t.nn.Linear(embedding_size, embedding_num, bias=False)
        t.nn.init.xavier_normal_(self.projection.weight)
        if share_weight:
            self.x_logit_scale = (embedding_size ** -0.5)
            self.projection.weight = self.embedding.word_embedding.weight
        else:
            self.x_logit_scale = 1

    def forward(self, input, decoder_input):
        input_encoded = self._encoder_forward(input)
        decoder_output = self._decoder_forward(input, input_encoded, decoder_input)
        output = self.projection(decoder_output) * self.x_logit_scale
        return output

    def _encoder_forward(self, input):
        encoder_pad_mask = get_pad_mask(input)
        encoder_self_attention_mask = get_self_attention_mask(input)
        input_embedding = self.embedding(input, encoder_pad_mask)
        input_encoded = self.encoder(input_embedding, encoder_self_attention_mask, encoder_pad_mask)
        return input_encoded

    def _decoder_forward(self, input, input_encoded, decoder_input):
        decoder_sequential_mask = get_sequential_mask(decoder_input, input)
        decoder_pad_mask = get_pad_mask(decoder_input)
        decoder_self_attention_mask = get_self_attention_mask(decoder_input)
        decoder_self_attention_mask *= decoder_sequential_mask
        decoder_dot_attention_mask = get_dot_attention_mask(input, decoder_input)
        decoder_input_embedding = self.embedding(decoder_input, decoder_pad_mask)
        decoder_output = self.decoder(decoder_input_embedding, input_encoded, input_encoded,
                                      decoder_self_attention_mask, decoder_dot_attention_mask, decoder_pad_mask)
        return decoder_output

    def greedy_search(self, input, max_decode_lenth, bos_id):
        input_encoded = self._encoder_forward(input)
        batch_size = input.size(0)
        device = input.device

        output_token = t.zeros((batch_size, max_decode_lenth), dtype=t.long, device=device)

        for i in range(max_decode_lenth):
            if i == 0:
                decoder_input = t.ones((batch_size, 1), dtype=t.long, device=device) * bos_id

            decoder_output = self._decoder_forward(input, input_encoded, decoder_input)
            decoder_projected = self.projection(decoder_output) * self.x_logit_scale
            last_token = t.argmax(decoder_projected, -1)[:, -1]
            output_token[:, i] = last_token
        return output_token

    def beam_search(self, input):
        pass




#
# import torch as t
# from Transformers.models import TED
# input = t.Tensor([[1,2,1,3,0],[1,2,1,0,0]]).long()
# target = t.Tensor([[1,1,0],[1,1,1]]).long()
# ted = TED(embedding_num=20, embedding_size=128, max_lenth=500, max_batch_size=64, dropout=0.1, hidden_size=128, num_head=4, num_step=6)
# ted(input, target)

#
#
#
