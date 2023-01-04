#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch09.html


第9章: RNN, CNN

--
以下を参考。
https://kakedashi-engineer.appspot.com/2020/05/14/nlp100-ch9/

"""

#################################################################################
# 80. ID番号への変換Permalink
# 問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
# 学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
# 学習データ中で2回以上出現する単語にID番号を付与せよ．
# そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
# ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．

base='06_dir/'

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 
test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

vectorizer = CountVectorizer(min_df=2)
train_title = train.iloc[:,1].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]

d = dict()

for i in range(len(words)):
    d[words[i]] = i+1
def get_id(sentence):
    r = []
    for word in sentence:
      r.append(d.get(word,0))
    return r

def df2id(df):
    ids = []
    for i in df.iloc[:,1].str.lower():
      ids.append(get_id(i.split()))
    return ids

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)



#################################################################################
# 81. RNNによる予測
import torch
dw = 300
dh = 50
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(len(words)+1,dw)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        y = self.softmax(y)
        return y




#################################################################################
# 82. 確率的勾配降下法による学習
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# %load_ext tensorboard
# !rm -rf ./runs
# %tensorboard --logdir ./runs
writer = SummaryWriter()

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        # y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()



train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 
test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)


model = RNN()
ds = TensorDataset(X_train, y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(2):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      y_pred = model(X_train)
      loss = loss_fn(y_pred, y_train) 
      writer.add_scalar('Loss/train', loss, epoch)
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid)
      loss = loss_fn(y_pred, y_valid)
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))





#################################################################################



import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# %load_ext tensorboard
# !rm -rf ./runs
# %tensorboard --logdir ./runs
writer = SummaryWriter()

max_len = 10
dw = 300
dh = 50
n_vocab = len(words) + 2
PAD = len(words) + 1

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:] # 最後のステップ
        y = self.linear(y)
        # y = self.softmax(y) # torch.nn.CrossEntropyLoss()がsoftmaxは含む
        return y

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [PAD] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

def accuracy(pred, label):
  pred = np.argmax(pred.data.to('cpu').numpy(), axis=1)
  label = label.data.to('cpu').numpy()
  return (pred == label).mean()

train = pd.read_csv(base+'train.txt', header=None, sep='\t') 
valid = pd.read_csv(base+'valid.txt', header=None, sep='\t') 


test = pd.read_csv(base+'test.txt', header=None, sep='\t') 

X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = np.loadtxt(base+'y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt(base+'y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt(base+'y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)


model = RNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
ds = TensorDataset(X_train.to(device), y_train.to(device))
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      y_pred = model(X_train.to(device))
      loss = loss_fn(y_pred, y_train.to(device)) 
      writer.add_scalar('Loss/train', loss, epoch)
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)
      print (accuracy(y_pred,y_train))

      y_pred = model(X_valid.to(device))
      loss = loss_fn(y_pred, y_valid.to(device))
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)
      print (accuracy(y_pred,y_valid))




#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################





#################################################################################


































