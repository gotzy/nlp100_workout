#!/usr/bin/env python
# coding: utf-8

# ### 80. ID番号への変換

# In[1]:


import re
import spacy
import torch


# In[2]:


nlp = spacy.load('en')
categories = ['b', 't', 'e', 'm']
category_names = ['business', 'science and technology', 'entertainment', 'health']


# In[3]:


def tokenize(x):
    x = re.sub(r'\s+', ' ', x)
    x = nlp.make_doc(x)
    x = [d.text for d in x]
    return x

def read_feature_dataset(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_t = [categories.index(line[0]) for line in dataset]
    dataset_x = [tokenize(line[1]) for line in dataset]
    return dataset_x, torch.tensor(dataset_t, dtype=torch.long)


# In[4]:


train_x, train_t = read_feature_dataset('data/train.txt')
valid_x, valid_t = read_feature_dataset('data/valid.txt')
test_x, test_t = read_feature_dataset('data/test.txt')


# In[5]:


from collections import Counter


# In[6]:


counter = Counter([
    x
    for sent in train_x
    for x in sent
])

vocab_in_train = [
    token
    for token, freq in counter.most_common()
    if freq > 1
]
len(vocab_in_train)


# 語彙を用意し，ID番号の列に変換できるようにする

# In[7]:


vocab_list = ['[UNK]'] + vocab_in_train
vocab_dict = {x:n for n, x in enumerate(vocab_list)}


# In[8]:


def sent_to_ids(sent):
    return torch.tensor([vocab_dict[x if x in vocab_dict else '[UNK]'] for x in sent], dtype=torch.long)


# In[9]:


print(train_x[0])
print(sent_to_ids(train_x[0]))


# ID番号の列に変換しておく

# In[10]:


def dataset_to_ids(dataset):
    return [sent_to_ids(x) for x in dataset]


# In[11]:


train_s = dataset_to_ids(train_x)
valid_s = dataset_to_ids(valid_x)
test_s = dataset_to_ids(test_x)


# In[12]:


train_s[:3]


# ### 81. RNNによる予測

# In[13]:


import random as rd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


# データの長さが可変長になったので，データセットのクラスも少し変わります

# In[14]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.lengths = torch.tensor([len(x) for x in source])
        self.size = len(source)
    
    def __len__(self):
        return self.size
            
    def __getitem__(self, index):
        return {
            'src':self.source[index],
            'trg':self.target[index],
            'lengths':self.lengths[index],
        }
    
    def collate(self, xs):
        return {
            'src':pad([x['src'] for x in xs]),
            'trg':torch.stack([x['trg'] for x in xs], dim=-1),
            'lengths':torch.stack([x['lengths'] for x in xs], dim=-1)
        }


# In[15]:


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, width, shuffle = False):
        self.dataset = dataset
        self.width = width
        self.shuffle = shuffle
        if not shuffle:
            self.indices = torch.arange(len(dataset))
            
    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))
        index = 0
        while index < len(self.dataset):
            yield self.indices[index : index + self.width]
            index += self.width


# バッチ内の系列の長さが降順になってるとうれしい

# In[16]:


class DescendingSampler(Sampler):
    def __init__(self, dataset, width, shuffle = False):
        assert not shuffle
        super().__init__(dataset, width, shuffle)
        self.indices = self.indices[self.dataset.lengths[self.indices].argsort(descending=True)]


# batchを事例数で積んでいくと無駄なパディングが多く発生するので，なるべく同じ長さのシーケンスをバッチに積んで，バッチのトークン数に基づいてバッチを切っていきます．

# In[17]:


class MaxTokensSampler(Sampler):
    def __iter__(self):
        self.indices = torch.randperm(len(self.dataset))
        self.indices = self.indices[self.dataset.lengths[self.indices].argsort(descending=True)]
        for batch in self.generate_batches():
            yield batch

    def generate_batches(self):
        batches = []
        batch = []
        acc = 0
        max_len = 0
        for index in self.indices:
            acc += 1
            this_len = self.dataset.lengths[index]
            max_len = max(max_len, this_len)
            if acc * max_len > self.width:
                batches.append(batch)
                batch = [index]
                acc = 1
                max_len = this_len
            else:
                batch.append(index)
        if batch != []:
            batches.append(batch)
        rd.shuffle(batches)
        return batches


# In[18]:


def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=8):
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler = sampler(dataset, width, shuffle),
        collate_fn = dataset.collate,
        num_workers = num_workers,
    )

def gen_descending_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler = DescendingSampler, shuffle = False, num_workers = num_workers)

def gen_maxtokens_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler = MaxTokensSampler, shuffle = True, num_workers = num_workers)


# データセットを用意する

# In[19]:


train_dataset = Dataset(train_s, train_t)
valid_dataset = Dataset(valid_s, valid_t)
test_dataset = Dataset(test_s, test_t)


# LSTMのモデル

# In[20]:


class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, num_layers = 1)
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)
        self.out.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, batch, h=None):
        x = self.embed(batch['src'])
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        h = self.out(h)
        return h.squeeze(0)


# In[21]:


model = LSTMClassifier(len(vocab_dict), 300, 50, 4)


# 予測する(問題文にあるsoftmaxはかけてない)

# In[22]:


loader = gen_loader(test_dataset, 10, DescendingSampler, False)
model(iter(loader).next()).argmax(dim=-1)


# ### 82. 確率的勾配降下法による学習

# TaskとTrainer

# In[23]:


class Task:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, model, batch):
        model.zero_grad()
        loss = self.criterion(model(batch), batch['trg'])
        loss.backward()
        return loss.item()
    
    def valid_step(self, model, batch):
        with torch.no_grad():
            loss = self.criterion(model(batch), batch['trg'])
        return loss.item()


# In[24]:


class Trainer:
    def __init__(self, model, loaders, task, optimizer, max_iter, device = None):
        self.model = model
        self.model.to(device)
        self.train_loader, self.valid_loader = loaders
        self.task = task
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.device = device
    
    def send(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch
        
    def train_epoch(self):
        self.model.train()
        acc = 0
        for n, batch in enumerate(self.train_loader):
            batch = self.send(batch)
            acc += self.task.train_step(self.model, batch)
            self.optimizer.step()
        return acc / n
            
    def valid_epoch(self):
        self.model.eval()
        acc = 0
        for n, batch in enumerate(self.valid_loader):
            batch = self.send(batch)
            acc += self.task.valid_step(self.model, batch)
        return acc / n
    
    def train(self):
        for epoch in range(self.max_iter):
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()
            print('epoch {}, train_loss:{:.5f}, valid_loss:{:.5f}'.format(epoch, train_loss, valid_loss))


# GPUはね，使う

# In[25]:


device = torch.device('cuda')
model = LSTMClassifier(len(vocab_dict), 300, 128, 4)
loaders = (
    gen_loader(train_dataset, 1),
    gen_loader(valid_dataset, 1),
)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
trainer = Trainer(model, loaders, task, optimizer, 3, device)
trainer.train()


# 予測もする

# In[26]:


import numpy as np


# In[27]:


class Predictor:
    def __init__(self, model, loader, device=None):
        self.model = model
        self.loader = loader
        self.device = device
        
    def send(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch
        
    def infer(self, batch):
        self.model.eval()
        batch = self.send(batch)
        return self.model(batch).argmax(dim=-1).item()
        
    def predict(self):
        lst = []
        for batch in self.loader:
            lst.append(self.infer(batch))
        return lst


# In[28]:


def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])


# In[29]:


predictor = Predictor(model, gen_loader(train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[30]:


predictor = Predictor(model, gen_loader(test_dataset, 1), device)
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 83. ミニバッチ化・GPU上での学習

# In[31]:


model = LSTMClassifier(len(vocab_dict), 300, 128, 4)
loaders = (
    gen_maxtokens_loader(train_dataset, 4000),
    gen_descending_loader(valid_dataset, 128),
)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, nesterov=True)
trainer = Trainer(model, loaders, task, optimizer, 10, device)
trainer.train()


# In[32]:


predictor = Predictor(model, gen_loader(train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[33]:


predictor = Predictor(model, gen_loader(test_dataset, 1), device)
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 84. 単語ベクトルの導入

# In[34]:


from gensim.models import KeyedVectors


# In[35]:


vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[36]:


def init_embed(embed):
    for i, token in enumerate(vocab_list):
        if token in vectors:
            embed.weight.data[i] = torch.from_numpy(vectors[token])
    return embed


# In[37]:


model = LSTMClassifier(len(vocab_dict), 300, 128, 4)
init_embed(model.embed)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True)
trainer = Trainer(model, loaders, task, optimizer, 10, device)
trainer.train()


# In[38]:


predictor = Predictor(model, gen_loader(train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[39]:


predictor = Predictor(model, gen_loader(test_dataset, 1), device)
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 85. 双方向RNN・多層化

# In[40]:


class BiLSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, num_layers = 2, bidirectional = True)
        self.out = nn.Linear(h_size * 2, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)
        nn.init.uniform_(self.out.weight, -0.1, 0.1)
    
    def forward(self, batch, h=None):
        x = self.embed(batch['src'])
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        h = h[-2:]
        h = h.transpose(0,1)
        h = h.contiguous().view(-1, h.size(1) * h.size(2))
        h = self.out(h)
        return h


# In[41]:


model = BiLSTMClassifier(len(vocab_dict), 300, 128, 4)
init_embed(model.embed)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True)
trainer = Trainer(model, loaders, task, optimizer, 10, device)
trainer.train()


# In[42]:


predictor = Predictor(model, gen_loader(train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[43]:


predictor = Predictor(model, gen_loader(test_dataset, 1), device)
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 86. 畳み込みニューラルネットワーク (CNN)

# PADのあるデータセットにする

# In[44]:


cnn_vocab_list = ['[PAD]', '[UNK]'] + vocab_in_train
cnn_vocab_dict = {x:n for n, x in enumerate(cnn_vocab_list)}


# In[45]:


def cnn_sent_to_ids(sent):
    return torch.tensor([cnn_vocab_dict[x if x in cnn_vocab_dict else '[UNK]'] for x in sent], dtype=torch.long)

print(train_x[0])
print(cnn_sent_to_ids(train_x[0]))


# In[46]:


def cnn_dataset_to_ids(dataset):
    return [cnn_sent_to_ids(x) for x in dataset]


# In[47]:


cnn_train_s = cnn_dataset_to_ids(train_x)
cnn_valid_s = cnn_dataset_to_ids(valid_x)
cnn_test_s = cnn_dataset_to_ids(test_x)


# In[48]:


cnn_train_s[:3]


# Datasetのパディングはpadを使わないことにしました

# In[49]:


class CNNDataset(Dataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        src = [torch.cat([x['src'], torch.zeros(max_seq_len - x['lengths'], dtype=torch.long)], dim=-1) for x in xs]
        src = torch.stack(src)
        mask = [[1] * x['lengths'] + [0] * (max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'src':src,
            'trg':torch.tensor([x['trg'] for x in xs]),
            'mask':mask,
        }


# In[50]:


cnn_train_dataset = CNNDataset(cnn_train_s, train_t)
cnn_valid_dataset = CNNDataset(cnn_valid_s, valid_t)
cnn_test_dataset = CNNDataset(cnn_test_s, test_t)


# CNNモデルをつくっていく

# In[51]:


class CNNClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.conv = nn.Conv1d(e_size, h_size, 3, padding=1)
        self.act = nn.ReLU()
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, 0, 0.1)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)
    
    def forward(self, batch):
        x = self.embed(batch['src'])
        x = self.dropout(x)
        x = self.conv(x.transpose(-1, -2))
        x = self.act(x)
        x = self.dropout(x)
        x.masked_fill_(batch['mask'].unsqueeze(-2) == 0, -1e4)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = self.out(x)
        return x


# ### 87. 確率的勾配降下法によるCNNの学習

# 学習させる

# In[52]:


def init_cnn_embed(embed):
    for i, token in enumerate(cnn_vocab_list):
        if token in vectors:
            embed.weight.data[i] = torch.from_numpy(vectors[token])
    return embed


# In[53]:


model = CNNClassifier(len(cnn_vocab_dict), 300, 128, 4)
init_cnn_embed(model.embed)
loaders = (
    gen_maxtokens_loader(cnn_train_dataset, 4000),
    gen_descending_loader(cnn_valid_dataset, 32),
)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
trainer = Trainer(model, loaders, task, optimizer, 10, device)
trainer.train()


# In[54]:


predictor = Predictor(model, gen_loader(cnn_train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[55]:


predictor = Predictor(model, gen_loader(cnn_test_dataset, 1), device)
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 88. パラメータチューニング

# In[56]:


class BiLSTMCNNDataset(Dataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        mask = [[1] * x['lengths'] + [0] * (max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'src':pad([x['src'] for x in xs]),
            'trg':torch.stack([x['trg'] for x in xs], dim=-1),
            'mask':mask,
            'lengths':torch.stack([x['lengths'] for x in xs], dim=-1)
        }


# In[57]:


rnncnn_train_dataset = BiLSTMCNNDataset(cnn_train_s, train_t)
rnncnn_valid_dataset = BiLSTMCNNDataset(cnn_valid_s, valid_t)
rnncnn_test_dataset = BiLSTMCNNDataset(cnn_test_s, test_t)


# In[58]:


class BiLSTMCNNClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, bidirectional = True)
        self.conv = nn.Conv1d(h_size* 2, h_size, 3, padding=1)
        self.act = nn.ReLU()
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)
    
    def forward(self, batch, h=None):
        x = self.embed(batch['src'])
        x = self.dropout(x)
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        x, _ = unpack(x)
        x = self.dropout(x)
        x = self.conv(x.permute(1, 2, 0))
        x = self.act(x)
        x = self.dropout(x)
        x.masked_fill_(batch['mask'].unsqueeze(-2) == 0, -1)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = self.out(x)
        return x


# In[59]:


loaders = (
    gen_maxtokens_loader(rnncnn_train_dataset, 4000),
    gen_descending_loader(rnncnn_valid_dataset, 32),
)
task = Task()
for h in [32, 64, 128, 256, 512]:
    model = BiLSTMCNNClassifier(len(cnn_vocab_dict), 300, h, 4)
    init_cnn_embed(model.embed)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)
    trainer = Trainer(model, loaders, task, optimizer, 10, device)
    trainer.train()
    predictor = Predictor(model, gen_loader(rnncnn_test_dataset, 1), device)
    pred = predictor.predict()
    print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 89. 事前学習済み言語モデルからの転移学習

# huggingface/transformers( https://github.com/huggingface/transformers )を使っていく

# In[60]:


from transformers import *


# In[61]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[62]:


def read_for_bert(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_t = [categories.index(line[0]) for line in dataset]
    dataset_x = [torch.tensor(tokenizer.encode(line[1]), dtype=torch.long) for line in dataset]
    return dataset_x, torch.tensor(dataset_t, dtype=torch.long)


# In[63]:


bert_train_x, bert_train_t = read_for_bert('data/train.txt')
bert_valid_x, bert_valid_t = read_for_bert('data/valid.txt')
bert_test_x, bert_test_t = read_for_bert('data/test.txt')


# In[64]:


class BertDataset(Dataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        src = [torch.cat([x['src'], torch.zeros(max_seq_len - x['lengths'], dtype=torch.long)], dim=-1) for x in xs]
        src = torch.stack(src)
        mask = [[1] * x['lengths'] + [0] * (max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'src':src,
            'trg':torch.tensor([x['trg'] for x in xs]),
            'mask':mask,
        }


# In[65]:


bert_train_dataset = BertDataset(bert_train_x, bert_train_t)
bert_valid_dataset = BertDataset(bert_valid_x, bert_valid_t)
bert_test_dataset = BertDataset(bert_test_x, bert_test_t)


# In[66]:


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-cased', num_labels=4)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-cased', config=config)
    
    def forward(self, batch):
        x = self.bert(batch['src'], attention_mask=batch['mask'])
        return x[0]


# In[67]:


model = BertClassifier()
loaders = (
    gen_maxtokens_loader(bert_train_dataset, 1000),
    gen_descending_loader(bert_valid_dataset, 32),
)
task = Task()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
trainer = Trainer(model, loaders, task, optimizer, 5, device)
trainer.train()


# In[68]:


predictor = Predictor(model, gen_loader(bert_train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[69]:


predictor = Predictor(model, gen_loader(bert_test_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(test_t, pred))

