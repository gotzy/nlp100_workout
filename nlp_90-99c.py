#!/usr/bin/env python
# coding: utf-8

# ### 90 データの準備

# In[1]:


get_ipython().system(' tar zxvf kftt-data-1.0.tar.gz')


# ginzaで形態素解析

# In[2]:


get_ipython().run_cell_magic('bash', '', "cat kftt-data-1.0/data/orig/kyoto-train.ja | sed 's/\\s+/ /g' | ginzame > train.ginza.ja\ncat kftt-data-1.0/data/orig/kyoto-dev.ja | sed 's/\\s+/ /g' | ginzame > dev.ginza.ja\ncat kftt-data-1.0/data/orig/kyoto-test.ja | sed 's/\\s+/ /g' | ginzame > test.ginza.ja\n")


# In[3]:


get_ipython().system(' head train.ginza.ja')


# spacyで単語分割

# In[4]:


import re
import spacy


# In[5]:


nlp = spacy.load('en')


# In[6]:


for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.en', 'train.spacy.en'),
    ('kftt-data-1.0/data/orig/kyoto-dev.en', 'dev.spacy.en'),
    ('kftt-data-1.0/data/orig/kyoto-test.en', 'test.spacy.en'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = nlp.make_doc(x)
            x = ' '.join([doc.text for doc in x])
            print(x, file=g)


# In[7]:


get_ipython().system(' head train.spacy.en')


# ginzaの解析結果から形態素に分割

# In[8]:


for src, dst in [
    ('train.ginza.ja', 'train.spacy.ja'),
    ('dev.ginza.ja', 'dev.spacy.ja'),
    ('test.ginza.ja', 'test.spacy.ja'),
]:
    with open(src) as f:
        lst = []
        tmp = []
        for x in f:
            x = x.strip()
            if x == 'EOS':
                lst.append(' '.join(tmp))
                tmp = []
            elif x != '':
                tmp.append(x.split('\t')[0])
    with open(dst, 'w') as f:
        for line in lst:
            print(line, file=f)


# In[9]:


get_ipython().system(' head train.spacy.ja')


# ### 91. 機械翻訳モデルの訓練

# In[10]:


get_ipython().system('fairseq-preprocess -s ja -t en      --trainpref train.spacy      --validpref dev.spacy      --destdir data91       --thresholdsrc 5      --thresholdtgt 5      --workers 20')


# In[12]:


get_ipython().system(' fairseq-train data91      --fp16      --save-dir save91      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --update-freq 1      --dropout 0.2 --weight-decay 0.0001      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 91.log')


# ### 92. 機械翻訳モデルの適用

# In[13]:


get_ipython().system("fairseq-interactive --path save91/checkpoint10.pt data91 < test.spacy.ja | grep '^H' | cut -f3 > 92.out")


# ### 93. BLEUスコアの計測

# In[14]:


get_ipython().system('fairseq-score --sys 92.out --ref test.spacy.en')


# ### 94. ビーム探索

# In[15]:


get_ipython().run_cell_magic('bash', '', "for N in `seq 1 20` ; do\n    fairseq-interactive --path save91/checkpoint10.pt --beam $N data91 < test.spacy.ja | grep '^H' | cut -f3 > 94.$N.out\ndone\n")


# In[16]:


get_ipython().run_cell_magic('bash', '', 'for N in `seq 1 20` ; do\n    fairseq-score --sys 94.$N.out --ref test.spacy.en > 94.$N.score\ndone\n')


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


def read_score(filename):
    with open(filename) as f:
        x = f.readlines()[1]
        x = re.search(r'(?<=BLEU4 = )\d*\.\d*(?=,)', x)
        return float(x.group())

xs = range(1, 21)
ys = [read_score(f'94.{x}.score') for x in xs]
plt.plot(xs, ys)
plt.show()


# ### 95. サブワード化

# In[19]:


import sentencepiece as spm


# In[20]:


spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')


# In[21]:


sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')


# In[22]:


for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.ja', 'train.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test.sub.ja'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)


# In[23]:


get_ipython().system(' head train.sub.ja')


# In[24]:


get_ipython().system(' subword-nmt learn-bpe -s 16000 < kftt-data-1.0/data/orig/kyoto-train.en > kyoto_en.codes')


# In[25]:


get_ipython().system(' subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-train.en > train.sub.en')
get_ipython().system(' subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-dev.en > dev.sub.en')
get_ipython().system(' subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-test.en > test.sub.en')


# In[26]:


get_ipython().system(' head train.sub.en')


# In[27]:


get_ipython().system('fairseq-preprocess -s ja -t en      --trainpref train.sub      --validpref dev.sub      --destdir data95       --workers 20')


# In[28]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save95      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --update-freq 1      --dropout 0.2 --weight-decay 0.0001      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 95.log')


# In[29]:


get_ipython().system("fairseq-interactive --path save95/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 95.out")


# In[30]:


def spacy_tokenize(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = ' '.join([doc.text for doc in nlp(x)])
            print(x, file=g)
spacy_tokenize('95.out', '95.out.spacy')


# In[31]:


get_ipython().system('fairseq-score --sys 95.out.spacy --ref test.spacy.en')


# In[32]:


get_ipython().run_cell_magic('bash', '', "for N in `seq 1 10` ; do\n    fairseq-interactive --path save95/checkpoint10.pt --beam $N data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 95.$N.out\ndone\n")


# In[33]:


for i in range(1, 11):
    spacy_tokenize(f'95.{i}.out', f'95.{i}.out.spacy')


# In[34]:


get_ipython().run_cell_magic('bash', '', 'for N in `seq 1 10` ; do\n    fairseq-score --sys 95.$N.out.spacy --ref test.spacy.en > 95.$N.score\ndone\n')


# In[35]:


xs = range(1, 11)
ys = [read_score(f'95.{x}.score') for x in xs]
plt.plot(xs, ys)
plt.show()


# ### 96. 学習過程の可視化

# In[37]:


get_ipython().system('fairseq-train data95      --fp16      --tensorboard-logdir log96      --save-dir save96      --max-epoch 5      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.2 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 96.log')


# pip install tensorboard tensorboardXをして，からtensorboradを起動してlocalhost:6006(など)にアクセスすると以下のような画面が出るはず
# 
# <img src='board.png'> 

# ### 97. ハイパー・パラメータの調整

# In[38]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save97_1      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.1 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 97_1.log')


# In[39]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save97_3      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.3 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 97_3.log')


# In[40]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save97_5      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.5 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 97_5.log')


# In[41]:


get_ipython().system("fairseq-interactive --path save97_1/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 97_1.out")
get_ipython().system("fairseq-interactive --path save97_3/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 97_3.out")
get_ipython().system("fairseq-interactive --path save97_5/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 97_5.out")


# In[42]:


spacy_tokenize('97_1.out', '97_1.out.spacy')
spacy_tokenize('97_3.out', '97_3.out.spacy')
spacy_tokenize('97_5.out', '97_5.out.spacy')


# In[43]:


get_ipython().system('fairseq-score --sys 97_1.out.spacy --ref test.spacy.en')
get_ipython().system('fairseq-score --sys 97_3.out.spacy --ref test.spacy.en')
get_ipython().system('fairseq-score --sys 97_5.out.spacy --ref test.spacy.en')


# In[44]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save97a      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.1 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 97a.log')


# In[50]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save97b      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 2e-4 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.1 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 97b.log')


# In[51]:


get_ipython().system('fairseq-train data95      --fp16      --save-dir save97c      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.1 --weight-decay 0.0001      --update-freq 1      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 97c.log')


# In[52]:


get_ipython().system("fairseq-interactive --path save97a/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 97a.out")
get_ipython().system("fairseq-interactive --path save97b/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 97b.out")
get_ipython().system("fairseq-interactive --path save97c/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 97c.out")


# In[53]:


spacy_tokenize('97a.out', '97a.out.spacy')
spacy_tokenize('97b.out', '97b.out.spacy')
spacy_tokenize('97c.out', '97c.out.spacy')


# In[54]:


get_ipython().system('fairseq-score --sys 97a.out.spacy --ref test.spacy.en')
get_ipython().system('fairseq-score --sys 97b.out.spacy --ref test.spacy.en')
get_ipython().system('fairseq-score --sys 97c.out.spacy --ref test.spacy.en')


# ### 98. ドメイン適応

# In[55]:


import tarfile


# In[56]:


with tarfile.open('en-ja.tar.gz') as tar:
    for f in tar.getmembers():
        if f.name.endswith('txt'):
            text = tar.extractfile(f).read().decode('utf-8')
            break


# In[57]:


data = text.splitlines()
data = [x.split('\t') for x in data]
data = [x for x in data if len(x) == 4]
data = [[x[3], x[2]] for x in data]


# In[58]:


with open('jparacrawl.ja', 'w') as f, open('jparacrawl.en', 'w') as g:
    for j, e in data:
        print(j, file=f)
        print(e, file=g)


# In[59]:


with open('jparacrawl.ja') as f, open('train.jparacrawl.ja', 'w') as g:
    for x in f:
        x = x.strip()
        x = re.sub(r'\s+', ' ', x)
        x = sp.encode_as_pieces(x)
        x = ' '.join(x)
        print(x, file=g)


# In[60]:


get_ipython().system(' subword-nmt apply-bpe -c kyoto_en.codes < jparacrawl.en > train.jparacrawl.en')


# In[61]:


get_ipython().system('fairseq-preprocess -s ja -t en      --trainpref train.jparacrawl      --validpref dev.sub      --destdir data98       --workers 20')


# In[63]:


get_ipython().system('fairseq-train data98      --fp16      --save-dir save98_1      --max-epoch 3      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000      --dropout 0.1 --weight-decay 0.0001      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 98_1.log')


# In[67]:


get_ipython().system("fairseq-interactive --path save98_1/checkpoint3.pt data98 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 98_1.out")


# In[68]:


spacy_tokenize('98_1.out', '98_1.out.spacy')


# In[69]:


get_ipython().system('fairseq-score --sys 98_1.out.spacy --ref test.spacy.en')


# In[70]:


get_ipython().system('fairseq-preprocess -s ja -t en      --trainpref train.sub      --validpref dev.sub      --tgtdict data98/dict.en.txt      --srcdict data98/dict.ja.txt      --destdir data98_2       --workers 20')


# In[72]:


get_ipython().system('fairseq-train data98_2      --fp16      --restore-file save98_1/checkpoint3.pt      --save-dir save98_2      --max-epoch 10      --arch transformer --share-decoder-input-output-embed      --optimizer adam --clip-norm 1.0      --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000      --dropout 0.1 --weight-decay 0.0001      --criterion label_smoothed_cross_entropy --label-smoothing 0.1      --max-tokens 8000 > 98_2.log')


# In[73]:


get_ipython().system("fairseq-interactive --path save98_2/checkpoint10.pt data98_2 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 98_2.out")


# In[74]:


spacy_tokenize('98_2.out', '98_2.out.spacy')


# In[75]:


get_ipython().system('fairseq-score --sys 98_2.out.spacy --ref test.spacy.en')


# ### 99. 翻訳サーバの構築

# ウェブアプリケーションの形式でやるべきなので，jupyter notebook上では無理です．
