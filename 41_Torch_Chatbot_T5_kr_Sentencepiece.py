!pip install sentencepiece

data_dir = "/content"

! pip list | grep sentencepiece

import sentencepiece as spm
import csv
import sys
import os
import random
import math
import re
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from sklearn.model_selection import train_test_split

from IPython.display import display
from tqdm import tqdm, tqdm_notebook, trange

# Setup seeds
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_LEN = 40
DECODER_LEN = ENCODER_LEN
BATCH_SIZE  = 128

N_EPOCHS = 20

import urllib3
import zipfile
import shutil
import pandas as pd

pd.set_option('display.max_colwidth', None)

http = urllib3.PoolManager()
url = 'https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv'
filename = 'ChatBotData.csv'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

train_data = pd.read_csv('ChatBotData.csv')
train_data.rename(columns={'Q': 'SRC', 'A': 'TRG'}, inplace=True)
train_data.head()

raw_src  = train_data['SRC']
raw_trg  = train_data['TRG']

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_data['SRC']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_data['TRG']))

# data를 저장할 폴더 입니다. 환경에 맞게 수정 하세요.
data_dir = "/content"

corpus = "corpus_src.txt"
prefix = "nmt_src_vocab"
vocab_size = 4000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK],<A>,<B>,<C>,<D>,<E>,<F>,<G>,<H>,<I>,<J>,<K>,<L>,<M>,<N>,<O>,<P>,<Q>,<R>,<S>,<T>,<U>,<V>,<W>,<X>,<Y>,<Z>") # 기타 추가 토큰

corpus = "corpus_trg.txt"
prefix = "nmt_trg_vocab"

vocab_size = 4000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK],<A>,<B>,<C>,<D>,<E>,<F>,<G>,<H>,<I>,<J>,<K>,<L>,<M>,<N>,<O>,<P>,<Q>,<R>,<S>,<T>,<U>,<V>,<W>,<X>,<Y>,<Z>") # 기타 추가 토큰

for f in os.listdir("."):
    print(f)

vocab_src_file = f"{data_dir}/nmt_src_vocab.model"
vocab_src = spm.SentencePieceProcessor()
vocab_src.load(vocab_src_file)

vocab_trg_file = f"{data_dir}/nmt_trg_vocab.model"
vocab_trg = spm.SentencePieceProcessor()
vocab_trg.load(vocab_trg_file)

lines = [
  "게임하고싶은데 할래?",
  "나 너 좋아하는 것 같아",
  "딥 러닝 자연어 처리를 잘 하고 싶어"
]
for line in lines:
    txt_2_tkn = vocab_src.encode_as_pieces(line)
    txt_2_ids = vocab_src.encode_as_ids(line)
    print(vocab_src.DecodeIds(txt_2_ids))
    print(vocab_src.DecodePieces(txt_2_tkn))

    ids2 = vocab_src.piece_to_id(txt_2_tkn)
    print(ids2)
    print(vocab_src.id_to_piece(ids2))
    print()
    print("Input     :", line)
    print("txt_2_tkn :", txt_2_tkn)
    print("txt_2_ids :", txt_2_ids)
    
train_df, test_df = train_test_split(train_data, test_size=0.2)

# 구분자 변경
train_df.to_csv('/content/ratings_train.txt', sep = '\t', index = False)
test_df.to_csv('/content/ratings_test.txt', sep = '\t', index = False)

train_df[:5]
test_df[:5]

""" train data 준비 """
def prepare_train(vocab_src, vocab_trg, infile, outfile):
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():

            src_document = row["SRC"]
            if type(src_document) != str:
                continue
            temp_src_sent = vocab_src.encode_as_pieces(src_document)
            if len(temp_src_sent)>256:
                temp_src_sent = temp_src_sent[:256]
            
            trg_document = row["TRG"]
            if type(trg_document) != str:
                continue
            temp_trg_sent = vocab_trg.encode_as_pieces(trg_document)
            if len(temp_trg_sent)>256:
                temp_trg_sent = temp_trg_sent[:256]

            instance = {"SRC": temp_src_sent, "TRG": temp_trg_sent }
            f.write(json.dumps(instance))
            f.write("\n")

prepare_train(vocab_src, vocab_trg, f"{data_dir}/ratings_train.txt", f"{data_dir}/ratings_train_t5.json")
prepare_train(vocab_src, vocab_trg, f"{data_dir}/ratings_test.txt",  f"{data_dir}/ratings_test_t5.json")
for f in os.listdir(data_dir):
    print(f)

data = [json.loads(line) for line in open('/content/ratings_train_t5.json', 'r')]
print(data[0])

max_len = 100

n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)
n_enc_seq = max_len            # json_encode_length
n_dec_seq = max_len            # json_decode_length
n_layers  = 2     # 6
hid_dim   = 256
pf_dim    = 1024
i_pad     = 0
n_heads   = 8
d_head    = 64
dropout   = 0.3
layer_norm_epsilon = 1e-12

n_output = n_dec_vocab

""" attention pad mask """
def create_padding_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return mask

""" attention decoder mask """
def create_look_ahead_mask(seq):
    look_ahead_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    look_ahead_mask = look_ahead_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return look_ahead_mask

""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (d_head ** 0.5)
        self.num_buckets = 32
        self.relative_attention_bias = torch.nn.Embedding(self.num_buckets, n_heads)
    
    def forward(self, Q, K, V, attn_mask, bidirectional=True):
        qlen, klen = Q.size(-2), K.size(-2)
        
        # (bs, n_heads, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        # (1, n_heads, n_q_seq, n_k_seq)
        position_bias = self.compute_bias(qlen, klen, bidirectional=bidirectional)
        scores += position_bias
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_heads, n_q_seq, n_k_seq)
        attention_weights = nn.Softmax(dim=-1)(scores)
        attention_weights = self.dropout(attention_weights)
        # (bs, n_heads, n_q_seq, d_v)
        output = torch.matmul(attention_weights, V)
        # (bs, n_heads, n_q_seq, d_v), (bs, n_heads, n_q_seq, n_v_seq)
        return output, attention_weights
    
    def compute_bias(self, qlen, klen, bidirectional=True):
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        # (qlen, klen)
        relative_position = memory_position - context_position
        # (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            num_buckets=self.num_buckets,
            bidirectional=bidirectional
        )
        # (qlen, klen)
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        # (qlen, klen, n_heads)
        values = self.relative_attention_bias(rp_bucket)
        # (1, n_heads, qlen, klen)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret


""" multi head attention """
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.q_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.k_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.v_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention()
        self.output_MHA = nn.Linear(n_heads * d_head, hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, attn_mask, bidirectional=False):
        batch_size = Q.size(0)
        
        # q : (bs, n_heads, n_q_seq, d_head)
        # k : (bs, n_heads, n_k_seq, d_head)
        # v : (bs, n_heads, n_v_seq, d_head)
        query = self.q_linear(Q).view(batch_size, -1, n_heads, d_head).transpose(1,2)
        key   = self.k_linear(K).view(batch_size, -1, n_heads, d_head).transpose(1,2)
        value = self.v_linear(V).view(batch_size, -1, n_heads, d_head).transpose(1,2)

        # (bs, n_heads, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # (bs, n_heads, n_q_seq, d_head), (bs, n_heads, n_q_seq, n_k_seq)
        scaled_attention, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask, bidirectional=bidirectional)
        
        # (bs, n_heads, n_q_seq, h_head * d_head)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_head)
        
        # (bs, n_heads, n_q_seq, e_embd)
        outputs = self.output_MHA(concat_attention)
        outputs = self.dropout(outputs)
        # (bs, n_q_seq, hid_dim), (bs, n_heads, n_q_seq, n_k_seq)
        return outputs, attn_prob

""" feed forward """
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.linear_1 = nn.Linear(hid_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, attention):
        output = self.linear_1(attention)
        output = F.relu(output)
        output = self.linear_2(output)
        return output

""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttentionLayer()
        self.ffn = PositionwiseFeedforwardLayer()
        
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, padding_mask):
        # (bs, n_enc_seq, hid_dim), (bs, n_heads, n_enc_seq, n_enc_seq)
        attention, attn_prob = self.attn(inputs, inputs, inputs, padding_mask)
        attention   = self.dropout1(attention)
        attention   = self.layernorm1(inputs + attention)  # (batch_size, input_seq_len, hid_dim)
        
        ffn_outputs = self.ffn(attention)  # (batch_size, input_seq_len, hid_dim)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attention + ffn_outputs)  # (batch_size, input_seq_len, hid_dim)

        # (bs, n_enc_seq, hid_dim), (bs, n_heads, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob

""" encoder """
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(n_enc_vocab, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self, inputs):
        # (bs, n_enc_seq, hid_dim)
        outputs = self.embedding(inputs)

        # (bs, n_enc_seq, n_enc_seq)
        attn_mask = create_padding_mask(inputs, inputs, i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, hid_dim), (bs, n_heads, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, hid_dim), [(bs, n_heads, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs

""" decoder layer """
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.attn   = MultiHeadAttentionLayer()
        self.attn_2 = MultiHeadAttentionLayer()

        self.ffn = PositionwiseFeedforwardLayer()

        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.layernorm3 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        # (bs, n_dec_seq, hid_dim), (bs, n_heads, n_dec_seq, n_dec_seq)
        attention1, self_attn_prob = self.attn(inputs, inputs, inputs, self_attn_mask, bidirectional=False)
        attention1 = self.dropout1(attention1)
        attention1 = self.layernorm1(inputs + attention1)
        
        # (bs, n_dec_seq, hid_dim), (bs, n_heads, n_dec_seq, n_enc_seq)
        attention2, dec_enc_attn_prob = self.attn_2(attention1, enc_outputs, enc_outputs, dec_enc_attn_mask)
        attention2 = self.dropout2(attention2)
        attention2 = self.layernorm2(attention1 + attention2)  # (batch_size, answerseq_len, hid_dim)

        ffn_outputs = self.ffn(attention2)  # (batch_size, answerseq_len, hid_dim)
        ffn_outputs = self.dropout3(ffn_outputs)
        ffn_outputs = self.layernorm3(attention2 + ffn_outputs)  # (batch_size, answerseq_len, hid_dim)
        
        # (bs, n_dec_seq, hid_dim), (bs, n_heads, n_dec_seq, n_dec_seq), (bs, n_heads, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob

""" decoder """
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(n_dec_vocab, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # (bs, n_dec_seq, hid_dim)

        dec_outputs = self.embedding(dec_inputs)

        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = create_padding_mask(dec_inputs, dec_inputs, i_pad)
        
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = create_look_ahead_mask(dec_inputs)
        
        # (bs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        
        # (bs, n_dec_seq, n_enc_seq)
        dec_enc_attn_mask = create_padding_mask(dec_inputs, enc_inputs, i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq, hid_dim), (bs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        # (bs, n_dec_seq, hid_dim), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]S
        return dec_outputs, self_attn_probs, dec_enc_attn_probs

""" t5 """
class T5(nn.Module):
    def __init__(self):
        super(T5, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.projection_lm = nn.Linear(hid_dim, n_enc_vocab, bias=False)
        # self.projection_lm.weight = self.embedding.weight
    
    def forward(self, enc_inputs, dec_inputs):
        
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)
        
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # (bs, n_dec_seq, n_enc_vocab)
        dec_outputs = self.projection_lm(dec_outputs)
        # (bs, n_dec_seq, n_enc_vocab), [(bs, n_heads, n_enc_seq, n_enc_seq)], [(bs, n_heads, n_dec_seq, n_dec_seq)], [(bs, n_heads, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    
    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]

""" Define Language Model Head """
class Language_Model_Head(nn.Module):
    def __init__(self):
        super().__init__()
        

        self.t5 = T5()
    
    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_dec_seq, n_enc_vocab), [(bs, n_heads, n_enc_seq, n_enc_seq)], [(bs, n_heads, n_dec_seq, n_dec_seq)], [(bs, n_heads, n_dec_seq, n_enc_seq)]
        logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.t5(enc_inputs, dec_inputs)
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

""" Language Model Dataset """
class Language_M_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab_src, vocab_trg, infile):
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.enc_inputs  = []
        self.dec_inputs  = []
        self.dec_outputs = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                instance = json.loads(line)
                enc_input = [vocab_src.piece_to_id(p) for p in instance["SRC"]]
                dec_input = [vocab_trg.piece_to_id(p) for p in instance["TRG"]]
                for _ in range(max_len-len(enc_input)):
                    enc_input.append(self.vocab_src.piece_to_id("[PAD]"))
                
                tmp_dec_input  = [vocab_trg.piece_to_id("[BOS]")] + dec_input
                for _ in range(max_len-len(tmp_dec_input)):
                    tmp_dec_input.append(self.vocab_trg.piece_to_id("[PAD]"))
                    
                tmp_dec_output = dec_input + [vocab_trg.piece_to_id("[EOS]")]
                for _ in range(max_len-len(tmp_dec_output)):
                    tmp_dec_output.append(self.vocab_trg.piece_to_id("[PAD]"))
                
                self.enc_inputs.append(enc_input)
                self.dec_inputs.append(tmp_dec_input)
                self.dec_outputs.append(tmp_dec_output)
                
    def __len__(self):
        assert len(self.dec_outputs) == len(self.enc_inputs)
        assert len(self.dec_outputs) == len(self.dec_inputs)
        return len(self.dec_outputs)
    
    def __getitem__(self, item):
        return (torch.tensor(self.enc_inputs[item]),
                torch.tensor(self.dec_inputs[item]),
                torch.tensor(self.dec_outputs[item]))

""" Language Model data collate_fn """
def L_M_collate(inputs):
    enc_inputs, dec_inputs, dec_outputs = list(zip(*inputs))

    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)
    dec_outputs = torch.nn.utils.rnn.pad_sequence(dec_outputs, batch_first=True, padding_value=0)

    batch = [
        enc_inputs,
        dec_inputs,
        dec_outputs
    ]
    return batch

""" 데이터 로더 """
batch_size = 8  #128
train_dataset = Language_M_Dataset(vocab_src, vocab_trg, f"{data_dir}/ratings_train_t5.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=L_M_collate)
test_dataset = Language_M_Dataset(vocab_src, vocab_trg, f"{data_dir}/ratings_test_t5.json")
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=L_M_collate)

print(train_dataset[0])

""" 모델 epoch 학습 """
def train_epoch(epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm_notebook(total=len(train_loader), desc=f"Train {epoch+1}") as pbar:
        for i, value in enumerate(train_loader):
            enc_inputs, dec_inputs, dec_outputs = map(lambda v: v.to(device), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]

            loss = criterion(logits.view(-1, logits.size(2)), dec_outputs.view(-1))

            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

learning_rate = 5e-5

model = Language_Model_Head()
model.to(device)

criterion_cls = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_epoch, best_loss, best_score = 0, 0, 0
losses, scores = [], []
for epoch in range(N_EPOCHS):
    loss = train_epoch(epoch, model, criterion_cls, optimizer, train_loader)
    # score = eval_epoch(model, test_loader)

    losses.append(loss)
    # scores.append(score)

    # if best_score < score:
    #     best_epoch, best_loss, best_score = epoch, loss, score
# print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")

"""
# table
data = {
    "loss": losses,
}
df = pd.DataFrame(data)
display(df)

# graph
plt.figure(figsize=[12, 4])
plt.plot(losses, label="loss")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
"""
