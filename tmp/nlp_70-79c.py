#!/usr/bin/env python
# coding: utf-8

# ### 70. 単語ベクトルの和による特徴量

# データを読み込んで単語に分割する

# In[1]:


import re
import spacy


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
    return dataset_x, dataset_t


# In[4]:


train_x, train_t = read_feature_dataset('data/train.txt')
valid_x, valid_t = read_feature_dataset('data/valid.txt')
test_x, test_t = read_feature_dataset('data/test.txt')


# 特徴ベクトルに変換する

# In[5]:


import torch
from gensim.models import KeyedVectors


# In[6]:


model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[7]:


def sent_to_vector(sent):
    lst = [torch.tensor(model[token]) for token in sent if token in model]
    return sum(lst) / len(lst)

def dataset_to_vector(dataset):
    return torch.stack([sent_to_vector(x) for x in dataset])


# In[8]:


train_v = dataset_to_vector(train_x)
valid_v = dataset_to_vector(valid_x)
test_v = dataset_to_vector(test_x)


# In[9]:


train_v[0]


# pickleにして保存

# In[10]:


import pickle


# In[11]:


train_t = torch.tensor(train_t).long()
valid_t = torch.tensor(valid_t).long()
test_t = torch.tensor(test_t).long()


# In[12]:


with open('data/train.feature.pickle', 'wb') as f:
    pickle.dump(train_v, f)
with open('data/train.label.pickle', 'wb') as f:
    pickle.dump(train_t, f)
    
with open('data/valid.feature.pickle', 'wb') as f:
    pickle.dump(valid_v, f)
with open('data/valid.label.pickle', 'wb') as f:
    pickle.dump(valid_t, f)
    
with open('data/test.feature.pickle', 'wb') as f:
    pickle.dump(test_v, f)
with open('data/test.label.pickle', 'wb') as f:
    pickle.dump(test_t, f)


# ### 71. 単層ニューラルネットワークによる予測

# 出題意図としては，単なる行列にsoftmaxかけたり，後々backwardしたりするというものなのかもしれませんが，ここではnn.Moduleを継承するクラスを作っていきます．

# In[13]:


import torch.nn as nn


# In[14]:


class Perceptron(nn.Module):
    def __init__(self, v_size, c_size):
        super().__init__()
        self.fc = nn.Linear(v_size, c_size, bias = False)
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, x):
        x = self.fc(x)
        return x


# In[15]:


model = Perceptron(300, 4)


# In[16]:


x = model(train_v[0])
x = torch.softmax(x, dim=-1)
x


# In[17]:


x = model(train_v[:4])
x = torch.softmax(x, dim=-1)
x


# ### 72. 損失と勾配の計算

# In[18]:


criterion = nn.CrossEntropyLoss()


# In[19]:


y = model(train_v[:1])
t = train_t[:1]
loss = criterion(y, t)
model.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配')
print(model.fc.weight.grad)


# In[20]:


model.zero_grad()
model.fc.weight.grad


# In[21]:


y = model(train_v[:4])
t = train_t[:4]
loss = criterion(y, t)
model.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配')
print(model.fc.weight.grad)


# ### 73. 確率的勾配降下法による学習

# 確率的勾配降下なので，データセットをシャッフルして少しずつ取り出すようにします

# データを持ってるのがDataset

# In[22]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, t):
        self.x = x
        self.t = t
        self.size = len(x)
    
    def __len__(self):
        return self.size
            
    def __getitem__(self, index):
        return {
            'x':self.x[index],
            't':self.t[index],
        }


# データセットをバッチに分割してバッチのインデックスを返すのがSampler

# In[23]:


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, width, shuffle=False):
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


# DatasetとSamplerをDataLoaderに渡すと，データセットをシャッフルして少しずつ取り出すことができる

# In[24]:


def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=8):
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler = sampler(dataset, width, shuffle),
        num_workers = num_workers,
    )


# データセットをバッチに積んでiterableにするの，人々が好きにいろんな方法でやっていて，どうやるのが適切なのかよくわかんない．
# 
# 個人的にはDatasetにデータ置いておいて，Samplerでインデックスをバッチに切り分けて，Loaderで回すっていうのが一番わかりやすいんじゃないかとは思っている　謎

# In[25]:


train_dataset = Dataset(train_v, train_t)
valid_dataset = Dataset(valid_v, valid_t)
test_dataset = Dataset(test_v, test_t)
loaders = (
    gen_loader(train_dataset, 1, shuffle = True),
    gen_loader(valid_dataset, 1),
)


# 損失を計算するTaskと最適化を回すTrainer

# In[26]:


import torch.optim as optim


# In[27]:


class Task:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, model, batch):
        model.zero_grad()
        loss = self.criterion(model(batch['x']), batch['t'])
        loss.backward()
        return loss.item()
    
    def valid_step(self, model, batch):
        with torch.no_grad():
            loss = self.criterion(model(batch['x']), batch['t'])
        return loss.item()


# In[28]:


class Trainer:
    def __init__(self, model, loaders, task, optimizer, max_iter, device = None):
        self.model = model
        self.model.to(device)
        self.train_loader, self.valid_loader = loaders
        self.task = task
        self.max_iter = max_iter
        self.optimizer = optimizer
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


# In[29]:


model = Perceptron(300, 4)
task = Task()
optimizer = optim.SGD(model.parameters(), 0.1)
trainer = Trainer(model, loaders, task, optimizer, 10)
trainer.train()


# ### 74. 正解率の計測

# In[30]:


import numpy as np


# In[31]:


class Predictor:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        
    def infer(self, batch):
        self.model.eval()
        return self.model(batch['x']).argmax(dim=-1).item()
        
    def predict(self):
        lst = []
        for batch in self.loader:
            lst.append(self.infer(batch))
        return lst


# In[32]:


def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])


# In[33]:


predictor = Predictor(model, gen_loader(train_dataset, 1))
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_t, pred))


# In[34]:


predictor = Predictor(model, gen_loader(test_dataset, 1))
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_t, pred))


# ### 75. 損失と正解率のプロット

# In[35]:


import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import clear_output


# In[36]:


class RealTimePlot:
    def __init__(self, legends):
        self.legends = legends
        self.fig, self.axs = plt.subplots(1, len(legends), figsize = (10, 5))
        self.lst = [[[] for _ in xs] for xs in legends]
        
    def __enter__(self):
        return self
        
    def update(self, *args):
        for i, ys in enumerate(args):
            for j, y in enumerate(ys):
                self.lst[i][j].append(y)
        clear_output(wait = True)
        for i, ax in enumerate(self.axs):
            ax.cla()
            for ys in self.lst[i]:
                ax.plot(ys)
            ax.legend(self.legends[i])
        display(self.fig)
        
    def __exit__(self, *exc_info):
        plt.close(self.fig)


# In[37]:


class VerboseTrainer(Trainer):
    def accuracy(self, true, pred):
        return np.mean([t == p for t, p in zip(true, pred)])
    
    def train(self, train_v, train_t, valid_v, valid_t):
        train_loader = gen_loader(Dataset(train_v, train_t), 1)
        valid_loader = gen_loader(Dataset(valid_v, valid_t), 1)
        with RealTimePlot([['学習', '検証']] * 2) as rtp:
            for epoch in range(self.max_iter):
                self.model.to(self.device)
                train_loss = self.train_epoch()
                valid_loss = self.valid_epoch()
                train_acc = self.accuracy(train_t, Predictor(self.model.cpu(), train_loader).predict())
                valid_acc = self.accuracy(valid_t, Predictor(self.model.cpu(), valid_loader).predict())
                rtp.update([train_loss, valid_loss], [train_acc, valid_acc])


# In[38]:


model = Perceptron(300, 4)
task = Task()
optimizer = optim.SGD(model.parameters(), 0.1)
trainer = VerboseTrainer(model, loaders, task, optimizer, 10)
train_predictor = Predictor(model, gen_loader(test_dataset, 1))
valid_predictor = Predictor(model, gen_loader(test_dataset, 1))
trainer.train(train_v, train_t, valid_v, valid_t)


# ### 76. チェックポイント

# 問題75から改変してもいいけど，めんどうなので73からやる

# In[39]:


import os


# In[40]:


class LoggingTrainer(Trainer):
    def save(self, epoch):
        torch.save({'epoch' : epoch, 'optimizer': self.optimizer}, f'trainer_states{epoch}.pt')
        torch.save(self.model.state_dict(), f'checkpoint{epoch}.pt')
    
    def train(self):
        for epoch in range(self.max_iter):
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()
            self.save(epoch)
            print('epoch {}, train_loss:{:.5f}, valid_loss:{:.5f}'.format(epoch, train_loss, valid_loss))


# In[41]:


model = Perceptron(300, 4)
task = Task()
optimizer = optim.SGD(model.parameters(), 0.1)
trainer = LoggingTrainer(model, loaders, task, optimizer, 10)
trainer.train()


# In[42]:


get_ipython().system(' ls result/checkpoint*')


# In[43]:


get_ipython().system(' ls result/trainer_states*')


# ### 77. ミニバッチ化

# In[44]:


from time import time
from contextlib import contextmanager


# In[45]:


@contextmanager
def timer(description):
    start = time()
    yield
    print(description, ': {:.3f} 秒'.format(time()-start))


# In[46]:


B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


# In[47]:


task = Task()
for b in B:
    model = Perceptron(300, 4)
    loaders = (
        gen_loader(train_dataset, b, shuffle = True),
        gen_loader(valid_dataset, 1)
    )
    optimizer = optim.SGD(model.parameters(), 0.1 * b)
    trainer = Trainer(model, loaders, task, optimizer, 3)
    with timer(f'バッチサイズ {b}'):
        trainer.train()


# ### 78. GPU上での学習

# In[48]:


device = torch.device('cuda')
model = Perceptron(300, 4)
task = Task()
loaders = (
    gen_loader(train_dataset, 128, shuffle = True),
    gen_loader(valid_dataset, 1),
)
optimizer = optim.SGD(model.parameters(), 0.1 * 128)
trainer = Trainer(model, loaders, task, optimizer, 3, device=device)
with timer('時間'):
    trainer.train()


# ### 79. 多層ニューラルネットワーク

# In[49]:


class ModelNLP79(nn.Module):
    def __init__(self, v_size, h_size, c_size):
        super().__init__()
        self.fc1 = nn.Linear(v_size, h_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(0.2)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# In[50]:


model = ModelNLP79(300, 128, 4)
task = Task()
loaders = (
    gen_loader(train_dataset, 128, shuffle = True),
    gen_loader(valid_dataset, 1)
)
optimizer = optim.SGD(model.parameters(), 0.1)
trainer = VerboseTrainer(model, loaders, task, optimizer, 30, device)
trainer.train(train_v, train_t, valid_v, valid_t)

