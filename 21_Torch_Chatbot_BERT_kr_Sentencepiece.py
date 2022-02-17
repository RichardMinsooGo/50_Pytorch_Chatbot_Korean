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
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # 기타 추가 토큰

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
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # 기타 추가 토큰

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

prepare_train(vocab_src, vocab_trg, f"{data_dir}/ratings_train.txt", f"{data_dir}/ratings_train.json")
prepare_train(vocab_src, vocab_trg, f"{data_dir}/ratings_test.txt", f"{data_dir}/ratings_test.json")
for f in os.listdir(data_dir):
    print(f)

data = [json.loads(line) for line in open('/content/ratings_train.json', 'r')]
print(data[0])

max_len = 100

n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)
n_enc_seq = max_len            # json_encode_length
n_seg_type = 2
n_layers  = 2
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

""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    """Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    query, key, value의 leading dimensions은 동일해야 합니다.
    key, value 에는 일치하는 끝에서 두 번째 차원이 있어야 합니다(예: seq_len_k = seq_len_v).
    MASK는 유형에 따라 모양이 다릅니다(패딩 혹은 미리보기(=look ahead)).
    그러나 추가하려면 브로드캐스트할 수 있어야 합니다.

    Args:
        query: query shape == (batch_size, n_heads, seq_len_q, depth)
        key: key shape     == (batch_size, n_heads, seq_len_k, depth)
        value: value shape == (batch_size, n_heads, seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (batch_size, n_heads, seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask):

        # Q와 K의 곱. 어텐션 스코어 행렬.
        matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

        # 스케일링
        # dk의 루트값으로 나눠준다.
        dk = key.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        # attention weight : (batch_size, n_heads, query의 문장 길이, key의 문장 길이)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # output : (batch_size, n_heads, query의 문장 길이, hid_dim/n_heads)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

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
    
    def forward(self, Q, K, V, attn_mask):
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
        scaled_attention, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask)
        
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
        self.pos_emb = nn.Embedding(n_enc_seq + 1, hid_dim)
        self.seg_emb = nn.Embedding(n_seg_type, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (bs, n_enc_seq, hid_dim)
        outputs = self.embedding(inputs) + self.pos_emb(positions)  + self.seg_emb(segments)

        # (bs, n_enc_seq, n_enc_seq)
        attn_mask = create_padding_mask(inputs, inputs, i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, hid_dim), (bs, n_heads, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, hid_dim), [(bs, n_heads, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs


# Model Define for Training
""" bert """
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        
        self.encoder = Encoder()

        self.linear = nn.Linear(hid_dim, hid_dim)
        self.activation = torch.tanh
    
    def forward(self, inputs, segments):
        # (bs, n_seq, hid_dim), [(bs, n_heads, n_enc_seq, n_enc_seq)]
        outputs, self_attn_probs = self.encoder(inputs, segments)
        # (bs, hid_dim)
        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)
        # (bs, n_enc_seq, n_enc_vocab), (bs, hid_dim), [(bs, n_heads, n_enc_seq, n_enc_seq)]
        return outputs, outputs_cls, self_attn_probs
    
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
        
        self.bert = BERT()
        # classfier
        self.projection_cls = nn.Linear(hid_dim, 2, bias=False)
        # lm
        self.projection_lm = nn.Linear(hid_dim, n_output, bias=False)
        self.projection_lm.weight = self.bert.encoder.embedding.weight
    
    def forward(self, inputs, segments):
        # (bs, n_enc_seq, hid_dim), (bs, hid_dim), [(bs, n_heads, n_enc_seq, n_enc_seq)]
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        # (bs, 2)
        logits_cls = self.projection_cls(outputs_cls)
        # (bs, n_enc_seq, n_enc_vocab)
        logits_lm = self.projection_lm(outputs)
        # (bs, n_enc_vocab), (bs, n_enc_seq, n_enc_vocab), [(bs, n_heads, n_enc_seq, n_enc_seq)]
        return logits_cls, logits_lm, attn_probs

""" Language Model Dataset """
class Language_M_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab_src, vocab_trg, infile):
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.src_sentences = []
        self.trg_sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc="Loading Dataset", unit=" lines")):
                data = json.loads(line)
                
                src_sentence = [self.vocab_src.piece_to_id("[CLS]")] + [self.vocab_src.piece_to_id(p) for p in data["SRC"]] + [self.vocab_src.piece_to_id("[SEP]")]
                
                trg_sentence = []
                trg_sentence = [0] * (len(src_sentence))
                segments = [0] * (len(src_sentence))

                trg_sentence_tmp = [self.vocab_trg.piece_to_id(p) for p in data["TRG"]] + [self.vocab_trg.piece_to_id("[SEP]")]
                
                # print("\n---src_sentence :",src_sentence)
                for i in trg_sentence_tmp:
                    src_sentence.append(self.vocab_src.piece_to_id("[MASK]"))
                    segments.append(1)
                
                for _ in range(max_len-len(src_sentence)):
                    src_sentence.append(self.vocab_src.piece_to_id("[MASK]"))
                    
                trg_sentence += [self.vocab_trg.piece_to_id(p) for p in data["TRG"]] + [self.vocab_trg.piece_to_id("[SEP]")]
                
                for _ in range(max_len - len(trg_sentence)):
                    trg_sentence.append(0)
                    segments.append(0)
                
                self.src_sentences.append(src_sentence)
                self.trg_sentences.append(trg_sentence)
                
                self.segments.append(segments)
    
    def __len__(self):
        assert len(self.src_sentences) == len(self.trg_sentences)
        assert len(self.src_sentences) == len(self.segments)
        return len(self.src_sentences)
    
    def __getitem__(self, item):
        return (torch.tensor(self.src_sentences[item]),
                torch.tensor(self.trg_sentences[item]),
                torch.tensor(self.segments[item]))

""" Language Model data collate_fn """
def L_M_collate(inputs):
    src_inputs, trg_outputs, segments = list(zip(*inputs))

    src_inputs  = torch.nn.utils.rnn.pad_sequence(src_inputs, batch_first=True, padding_value=0)
    trg_outputs = torch.nn.utils.rnn.pad_sequence(trg_outputs, batch_first=True, padding_value=0)
    segments    = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    # src_inputs, trg_outputs  = torch.nn.utils.rnn.pad_sequence([src_inputs,trg_outputs], batch_first=True, padding_value=0)

    batch = [
        src_inputs,
        trg_outputs,
        segments
    ]
    return batch

""" 데이터 로더 """
batch_size = 64  #128
train_dataset = Language_M_Dataset(vocab_src, vocab_trg, f"{data_dir}/ratings_train.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=L_M_collate)
test_dataset = Language_M_Dataset(vocab_src, vocab_trg, f"{data_dir}/ratings_test.json")
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=L_M_collate)

print(train_dataset[0])

""" 모델 epoch 학습 """
def train_epoch(epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm_notebook(total=len(train_loader), desc=f"Train {epoch+1}") as pbar:
        for i, value in enumerate(train_loader):
            src_inputs, trg_outputs, segments = map(lambda v: v.to(device), value)

            optimizer.zero_grad()
            outputs = model(src_inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]
            
            labels_lm = trg_outputs.contiguous()
            
            loss_lm = criterion(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            
            
            # loss = loss_lm
            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss_lm.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

learning_rate = 5e-5

model = Language_Model_Head()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_epoch, best_loss, best_score = 0, 0, 0
losses, scores = [], []
for epoch in range(N_EPOCHS):
    loss = train_epoch(epoch, model, criterion, optimizer, train_loader)
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
