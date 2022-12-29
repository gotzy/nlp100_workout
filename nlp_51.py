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
51. 特徴量抽出Permalink
学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

"""



import pandas as pd
import random


df01 = pd.read_csv( '06_dir/newsCorpora.csv' ,sep='\t', header=None)
df01.columns='ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP'.split(' \t ')
df02=df01[df01['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df02=df02.reset_index(drop=True)

ls02=list(df02.index)
random.seed(100)
random.shuffle(ls02)
df03=df02.iloc[ls02,:]
df03=df03.reset_index(drop=True)

df03b=df03.copy() # 50との違い

df04=df03b.iloc[:1334*8, :]
df05=df03b.iloc[1334*8:1334*9, :]
df06=df03b.iloc[1334*9:, :]





# def f50(df_,fname_):
#     with open(fname_, 'w') as f:
#         # print(colname, file=f)
#         for i in range(df_.shape[0]):
#             res=[ str(x) for x in df_.iloc[i,:].values.tolist()]
#             res2='\t'.join(res)
#             res3=res2.replace('\n','')   ### 改行コードを除去
#             print(res3, file=f)

# f50( df04 , '06_dir/train.txt' )
# f50( df05 , '06_dir/valid.txt' )
# f50( df06 , '06_dir/test.txt' )



## unix command
# wc -l train.txt valid.txt test.txt
#    10672 train.txt
#     1334 valid.txt
#     1334 test.txt
#    13340 total



## CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)

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










