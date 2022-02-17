import os
import torch.optim as optim
import random
import math
import re
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.legacy import data, datasets

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

#https://github.com/lovit/soynlp
!pip install soynlp

from soynlp.tokenizer import LTokenizer

tokenizer = LTokenizer()

tokenizer("내일 역 앞의 식당에서 밥 먹으러 나갈래 ?")

VOCAB_SIZE = 256

SRC_tokenizer = data.Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenizer,
    batch_first=True, init_token="<SOS>", eos_token="<EOS>", fix_length=ENCODER_LEN)

TRG_tokenizer = data.Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenizer,
    batch_first=True, init_token="<SOS>", eos_token="<EOS>", fix_length=DECODER_LEN)

trainset = data.TabularDataset(
        path='ChatBotData.csv', format='csv', skip_header=False,
        fields=[('SRC', SRC_tokenizer),('TRG', TRG_tokenizer)])

print(vars(trainset[2]))

print('훈련 샘플의 개수 : {}'.format(len(trainset)))

SRC_tokenizer.build_vocab(trainset.SRC, trainset.TRG, min_freq = 2) # 단어 집합 생성
TRG_tokenizer.vocab = SRC_tokenizer.vocab# 단어 집합 생성

PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = SRC_tokenizer.vocab.stoi['<pad>'], SRC_tokenizer.vocab.stoi['<SOS>'], SRC_tokenizer.vocab.stoi['<EOS>'], SRC_tokenizer.vocab.stoi['<unk>']

# Difine HyperParameter
n_enc_vocab = len(SRC_tokenizer.vocab)
n_dec_vocab = len(TRG_tokenizer.vocab)

print('Encoder 단어 집합의 크기 :',n_enc_vocab)
print('Decoder 단어 집합의 크기 :',n_dec_vocab)

# Define Iterator
# dataloader batch has text and target item

dataloader = data.BucketIterator(
        trainset, batch_size=BATCH_SIZE,
        shuffle=True, repeat=False, sort=False, device = device)

# Hyper-parameters
n_layers  = 2     # 6
hid_dim   = 256
pf_dim    = 1024
n_heads   = 8
dropout   = 0.3
pe_source = 512
pe_target = 512
layer_norm_epsilon = 1e-12


""" attention pad mask """
def create_padding_mask(x):
    input_pad = 0
    mask = (x == input_pad).float()
    mask = mask.unsqueeze(1).unsqueeze(1)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask

""" attention decoder mask """
def create_look_ahead_mask(seq):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = seq.shape[1]
    look_ahead_mask = torch.ones(seq_len, seq_len)
    
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1).to(device)
    # padding_mask = create_padding_mask(seq).to(device) # 패딩 마스크도 포함
    # return torch.maximum(look_ahead_mask, padding_mask)

    return look_ahead_mask

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
        self.dropout = nn.Dropout(0.3)
        self.num_buckets = 32
        self.relative_attention_bias = torch.nn.Embedding(self.num_buckets, n_heads)
        
    def forward(self, query, key, value, mask, bidirectional=True):
        qlen, klen = query.size(-2), key.size(-2)
        # Q와 K의 곱. 어텐션 스코어 행렬.
        matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

        # 스케일링
        # dk의 루트값으로 나눠준다.
        dk = key.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        position_bias = self.compute_bias(qlen, klen, bidirectional=bidirectional)
        scaled_attention_logits += position_bias
        
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
        # (qlen, klen, n_head)
        values = self.relative_attention_bias(rp_bucket)
        # (1, n_head, qlen, klen)
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
    
    def __init__(self, hid_dim, n_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        assert hid_dim % self.n_heads == 0
        self.hid_dim = hid_dim
        
        # hid_dim을 n_heads로 나눈 값.
        self.depth = int(hid_dim/self.n_heads)
        
        # WQ, WK, WV에 해당하는 밀집층 정의
        self.q_linear = nn.Linear(hid_dim, hid_dim)
        self.k_linear = nn.Linear(hid_dim, hid_dim)
        self.v_linear = nn.Linear(hid_dim, hid_dim)

        self.scaled_dot_attn = ScaledDotProductAttention()
        
        # WO에 해당하는 밀집층 정의
        self.out = nn.Linear(hid_dim, hid_dim)

    # n_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = torch.reshape(
            inputs, (batch_size, -1, self.n_heads, self.depth))
        return torch.transpose(inputs, 1,2)

    def forward(self, inputs, bidirectional=False):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.shape[0]
        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, hid_dim)
        # k : (batch_size, key의 문장 길이, hid_dim)
        # v : (batch_size, value의 문장 길이, hid_dim)
        query = self.q_linear(query)
        key   = self.k_linear(key)
        value = self.v_linear(value)
        
        # 2. 헤드 나누기
        # q : (batch_size, n_heads, query의 문장 길이, hid_dim/n_heads)
        # k : (batch_size, n_heads, key의 문장 길이,   hid_dim/n_heads)
        # v : (batch_size, n_heads, value의 문장 길이, hid_dim/n_heads)
        query = self.split_heads(query, batch_size)
        key   = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, n_heads, query의 문장 길이, hid_dim/n_heads)
        # scaled_attention, _ = ScaledDotProductAttention(query, key, value, mask)
        scaled_attention, _ = self.scaled_dot_attn(
            query, key, value, mask, bidirectional = bidirectional)
        
        # (batch_size, query의 문장 길이, n_heads, hid_dim/n_heads)
        scaled_attention = torch.transpose(scaled_attention, 1,2)
        
        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, hid_dim)
        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.hid_dim))
        
        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, hid_dim)
        outputs = self.out(concat_attention)

        return outputs

""" feed forward """
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim):
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
        
        self.attn = MultiHeadAttentionLayer(hid_dim, n_heads)
        self.ffn = PositionwiseFeedforwardLayer(hid_dim, pf_dim)
        
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, padding_mask):
        attention   = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask}, bidirectional=False)
        attention   = self.dropout1(attention)
        attention   = self.layernorm1(inputs + attention)  # (batch_size, input_seq_len, hid_dim)
        
        ffn_outputs = self.ffn(attention)  # (batch_size, input_seq_len, hid_dim)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attention + ffn_outputs)  # (batch_size, input_seq_len, hid_dim)

        return ffn_outputs

""" encoder """
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.embedding    = nn.Embedding(n_enc_vocab, hid_dim)

        self.enc_layers   = EncoderLayer()
        self.dropout1     = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        emb = self.embedding(x)  # (batch_size, input_seq_len, hid_dim)
        output = self.dropout1(emb)

        for i in range(n_layers):
            output = self.enc_layers(output, padding_mask)

        return output  # (batch_size, input_seq_len, hid_dim)
    
""" decoder layer """
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.attn   = MultiHeadAttentionLayer(hid_dim, n_heads)
        self.attn_2 = MultiHeadAttentionLayer(hid_dim, n_heads)

        self.ffn = PositionwiseFeedforwardLayer(hid_dim, pf_dim)

        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.layernorm3 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
        attention1 = self.attn(
            {'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask}, bidirectional=False)
        attention1 = self.dropout1(attention1)
        attention1 = self.layernorm1(inputs + attention1)

        attention2 = self.attn_2(
            {'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask}, bidirectional=False)
        attention2 = self.dropout2(attention2)
        attention2 = self.layernorm2(attention1 + attention2)  # (batch_size, target_seq_len, hid_dim)

        ffn_outputs = self.ffn(attention2)  # (batch_size, target_seq_len, hid_dim)
        ffn_outputs = self.dropout3(ffn_outputs)
        ffn_outputs = self.layernorm3(attention2 + ffn_outputs)  # (batch_size, target_seq_len, hid_dim)

        return ffn_outputs  

""" decoder """
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.embedding    = nn.Embedding(n_dec_vocab, hid_dim)
        self.dec_layers = DecoderLayer()
        self.dropout      = nn.Dropout(dropout)
        
    def forward(self, enc_output, dec_input, padding_mask, look_ahead_mask):
        emb = self.embedding(dec_input)
        output = self.dropout(emb)
        for i in range(n_layers):
            output = self.dec_layers(output, enc_output, padding_mask, look_ahead_mask)

        return output
    
""" transformer """
class Transformer(nn.Module):
    def __init__(self, n_enc_vocab, n_dec_vocab,
                 n_layers, pf_dim, hid_dim, n_heads,
                 pe_source, pe_target, dropout):
        super(Transformer, self).__init__()
        
        # Ecoder and Decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fin_output = nn.Linear(hid_dim, n_dec_vocab)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, enc_inputs, dec_inputs):

        enc_padding_mask = create_padding_mask(enc_inputs)
        dec_padding_mask = create_padding_mask(enc_inputs)
        look_ahead_mask  = create_look_ahead_mask(dec_inputs)
        dec_target_padding_mask = create_padding_mask(dec_inputs).to(device) # 패딩 마스크도 포함
        look_ahead_mask  = torch.maximum(dec_target_padding_mask, look_ahead_mask)

        enc_output = self.encoder(enc_inputs, enc_padding_mask)
        dec_output = self.decoder(enc_output, dec_inputs, dec_padding_mask, look_ahead_mask)
        final_output = self.fin_output(dec_output)
        return final_output

# 모델 구축
model = Transformer(
    n_enc_vocab = n_enc_vocab,
    n_dec_vocab = n_dec_vocab,
    n_layers  = n_layers,
    pf_dim      = pf_dim,
    hid_dim     = hid_dim,
    n_heads     = n_heads,
    pe_source   = 512,
    pe_target   = 512,
    dropout     = dropout)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 네트워크 초기화
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# TransformerBlock모듈의 초기화 설정
model.apply(initialize_weights)

import os.path

if os.path.isfile('./checkpoints/transformermodel.pt'):
    model.load_state_dict(torch.load('./checkpoints/transformermodel.pt'))

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
# learning_rate = 2e-4
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

from IPython.display import clear_output
import datetime

Model_start_time = time.time()

# 학습 정의
def train(epoch, model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    with tqdm_notebook(total=len(dataloader), desc=f"Train {epoch+1}") as pbar:    
        for batch in dataloader:
            src_inputs = batch.SRC.to(device)
            trg_outputs = batch.TRG.to(device)

            with torch.set_grad_enabled(True):
                # Transformer에 입력
                logits_lm = model(src_inputs, trg_outputs)

                pad = torch.LongTensor(trg_outputs.size(0), 1).fill_(PAD_TOKEN).to(device)
                preds_id = torch.transpose(logits_lm,1,2)
                labels_lm = torch.cat((trg_outputs[:, 1:], pad), -1)

                optimizer.zero_grad()
                loss = criterion(preds_id, labels_lm)  # loss 계산
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss +=loss.item()

            pbar.update(1)
            # pbar.set_postfix_str(f"Loss {epoch_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # pbar.set_postfix_str(f"Loss {loss.result():.4f}")
            
    return epoch_loss / len(dataloader)

CLIP = 0.5

epoch_ = []
epoch_train_loss = []
# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True
# epoch 루프
best_epoch_loss = float("inf")

for epoch in range(N_EPOCHS):
    
    train_loss = train(epoch, model, dataloader, optimizer, criterion, CLIP)
    
    if train_loss < best_epoch_loss:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        best_epoch_loss = train_loss
        torch.save(model.state_dict(), './checkpoints/transformermodel.pt')

    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    
    # print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, N_EPOCHS, epoch_loss))
    # clear_output(wait = True)

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()

# Predict the trained model
trained_model = Transformer(
    n_enc_vocab = n_enc_vocab,
    n_dec_vocab = n_dec_vocab,
    n_layers  = n_layers,
    pf_dim      = pf_dim,
    hid_dim     = hid_dim,
    n_heads     = n_heads,
    pe_source    = 512,
    pe_target   = 512,
    dropout     = dropout).to(device)
trained_model.load_state_dict(torch.load('./checkpoints/transformermodel.pt'))

def stoi(vocab, token, max_len):
    #
    indices=[]
    token.extend(['<pad>'] * (max_len - len(token)))
    for string in token:
        if string in vocab:
            i = vocab.index(string)
        else:
            i = 0
        indices.append(i)
    return torch.LongTensor(indices).unsqueeze(0)

def itos(vocab, indices):
    text = []
    for i in indices.cpu()[0]:
        if i==1:
            break
        else:
            if i not in [PAD_TOKEN, START_TOKEN, END_TOKEN]:
                if i != UNK_TOKEN:
                    text.append(vocab[i])
                else:
                    text.append('??')
    return " ".join(text)

def evaluate(text):
    tokenizer = LTokenizer()
    token = tokenizer(text)
    input = stoi(SRC_tokenizer.vocab.itos, token, ENCODER_LEN).to(device)
    output = torch.LongTensor(1, 1).fill_(START_TOKEN).to(device)
    
    for i in range(DECODER_LEN):
        predictions = trained_model(input, output)
        predictions = predictions[:, -1:, :]
                            
        # PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions[:,:,3:], axis=-1) + 3
        if predicted_id == END_TOKEN:
            predicted_id = predicted_id
            break
        output = torch.cat((output, predicted_id),-1)
    return output

def predict(text):
    prediction = evaluate(text)
    predicted_sentence = itos(TRG_tokenizer.vocab.itos, prediction)
    
    return predicted_sentence

for idx in (11, 21, 31, 41, 51):
    print("Input        :", raw_src[idx])
    print("Prediction   :", predict(str(raw_src[idx])))
    print("Ground Truth :", raw_trg[idx],"\n")
    