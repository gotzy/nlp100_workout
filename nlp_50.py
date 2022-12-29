#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch06.html


第6章: 機械学習
本章では，Fabio Gasparetti氏が公開しているNews Aggregator Data Setを用い，
ニュース記事の見出しを
「ビジネス」「科学技術」「エンターテイメント」「健康」のカテゴリに
分類するタスク（カテゴリ分類）に取り組む．


---

https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
をダウンロードし、以下のように格納した。
./06_dir/NewsAggregatorDataset.zip
この前提で進める。

---
50. データの入手・整形
News Aggregator Data Setをダウンロードし、
以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
2. 情報源（publisher）が
    ”Reuters”, 
    “Huffington Post”, 
    “Businessweek”, 
    “Contactmusic.com”, 
    “Daily Mail”
    の事例（記事）のみを抽出する．
3. 抽出された事例をランダムに並び替える．
4. 抽出された事例の80%を学習データ，
   残りの10%ずつを
   検証データと評価データに分割し，
   それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
   ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ
   （このファイルは後に問題70で再利用する）．
   
学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

"""


############################################################################
# 1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
## unix command
# unzip NewsAggregatorDataset.zip 


############################################################################
# 2. 情報源（publisher）が
#     ”Reuters”, 
#     “Huffington Post”, 
#     “Businessweek”, 
#     “Contactmusic.com”, 
#     “Daily Mail”
#     の事例（記事）のみを抽出する．

import pandas as pd

df01 = pd.read_csv( '06_dir/newsCorpora.csv' ,sep='\t', header=None)

df01.columns='ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP'.split(' \t ')

df02=df01[df01['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]

df02=df02.reset_index(drop=True)

############################################################################
# 3. 抽出された事例をランダムに並び替える．

import random

ls02=list(df02.index)
random.seed(100)
random.shuffle(ls02)

df03=df02.iloc[ls02,:]
df03=df03.reset_index(drop=True)


############################################################################
# 4. 抽出された事例の80%を学習データ，
#    残りの10%ずつを
#    検証データと評価データに分割し，
#    それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
#    ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ
#    （このファイルは後に問題70で再利用する）．

############################################################################
# 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．


print(df03.shape[0])
# 13340

df03b=df03[['CATEGORY','TITLE']]

# train
df04=df03b.iloc[:1334*8, :]
print(df04.shape[0])

# valid
df05=df03b.iloc[1334*8:1334*9, :]
print(df05.shape[0])

# test
df06=df03b.iloc[1334*9:, :]
print(df06.shape[0])



# colname='\t'.join(df04.columns)

def f50(df_,fname_):
    with open(fname_, 'w') as f:
        # print(colname, file=f)
        for i in range(df_.shape[0]):
            res=[ str(x) for x in df_.iloc[i,:].values.tolist()]
            res2='\t'.join(res)
            res3=res2.replace('\n','')   ### 改行コードを除去
            print(res3, file=f)

f50( df04 , '06_dir/train.txt' )
f50( df05 , '06_dir/valid.txt' )
f50( df06 , '06_dir/test.txt' )



## unix command
# wc -l train.txt valid.txt test.txt
#    10672 train.txt
#     1334 valid.txt
#     1334 test.txt
#    13340 total


# awk '{print $1}' train.txt| sort| uniq -c 
# 4478 b
# 4226 e
#  741 m
# 1227 t


# awk '{print $1}' test.txt| sort| uniq -c
#  583 b
#  528 e
#   79 m
#  144 t

# awk '{print $1}' valid.txt| sort| uniq -c
#  566 b
#  525 e
#   90 m
#  153 t






