#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch07.html


第7章: 単語ベクトル
単語の意味を実ベクトルで表現する単語ベクトル（単語埋め込み）に関して，以下の処理を行うプログラムを作成せよ．

---

「GoogleNews-vectors-negative300.bin.gz」（1.5G）
をダウンロードし、
gunzip GoogleNews-vectors-negative300.bin.gz 
で解凍し、以下のように格納した。

https://qiita.com/fuwasegu/items/115b6f93bccb2115086e

./07_dir/GoogleNews-vectors-negative300.bin
この前提で進める。

"""

#################################################################################
# 60. 単語ベクトルの読み込みと表示
# Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル（300万単語・フレーズ，300次元）をダウンロードし，
# ”United States”の単語ベクトルを表示せよ．
# ただし，”United States”は内部的には”United_States”と表現されていることに注意せよ．

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('07_dir/GoogleNews-vectors-negative300.bin', binary=True)
print(model['United_States'])


#################################################################################
# 61. 単語の類似度
# “United States”と”U.S.”のコサイン類似度を計算せよ．

print( model.similarity( 'United_States', 'U.S.' ) )



#################################################################################
# 62. 類似度の高い単語10件
# “United States”とコサイン類似度が高い10語と，その類似度を出力せよ．

print( model.most_similar(positive=['United_States'],topn=10) )



#################################################################################
# 63. 加法構成性によるアナロジー
# “Spain”の単語ベクトルから”Madrid”のベクトルを引き，
# ”Athens”のベクトルを足したベクトルを計算し，
# そのベクトルと類似度の高い10語とその類似度を出力せよ．

print( model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'] ,topn=10) )



#################################################################################
# 64. アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし，
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．

import datetime
t1=datetime.datetime.now()

import pandas as pd
import numpy as np

# df01 = pd.read_csv('07_dir/questions-words.txt', delimiter=' ', header=None, index_col=None, skiprows=[0])

with open('07_dir/questions-words.txt') as f:
    line=f.readlines()

mark=''
ls00=list()
for x in line:
    if x.startswith(':'):
        mark=x
    else:
        res0 = [mark] + x.split()
        ls00.append(res0)
        
df01=pd.DataFrame(ls00)

##

ls01=list()

for  ii, (w1,w2,w3) in enumerate(zip(df01.iloc[:,1], df01.iloc[:,2], df01.iloc[:,3])):
    try:
        res1=model.most_similar(positive=[w2, w3], negative=[w1] ,topn=1)[0]
    except:
        res1=(None,None)
        print('except', ii, df01.iloc[ii,0])
    ls01.append(res1)
    

df01['result_64_w'] = [ x[0] for x in ls01 ]
df01['result_64_s'] = [ x[1] for x in ls01 ]


t2=datetime.datetime.now()
print(t2-t1)


# import pickle
# with open('07_dir/df01.pickle', 'wb') as f:
#     pickle.dump(df01, f)

# with open('07_dir/df01.pickle', 'rb') as f:
#     df01 = pickle.load(f)


#################################################################################
# 65. アナロジータスクでの正解率
# 64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．

semantic_analogy = [df01.iloc[i,4]==df01.iloc[i,5] for i in range(df01.shape[0])  if not df01.iloc[i,0].startswith(': gram')]
syntactic_analogy = [df01.iloc[i,4]==df01.iloc[i,5] for i in range(df01.shape[0])  if df01.iloc[i,0].startswith(': gram')]

acc = np.mean(semantic_analogy)
print('意味的アナロジー　正解率:', acc)

acc = np.mean(syntactic_analogy)
print('文法的アナロジー　正解率:', acc)


#################################################################################
# 66. WordSimilarity-353での評価
# The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
# 単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

import zipfile

# zipファイルから読み込む
with zipfile.ZipFile('07_dir/wordsim353.zip') as f:
    with f.open('combined.csv') as g:
        data = g.read()

# バイト列をデコード
data = data.decode('UTF-8').splitlines()
data = data[1:]

data = [line.split(',') for line in data]


for i, lst in enumerate(data):
    sim = model.similarity(lst[0], lst[1])
    data[i].append(sim)

df66=pd.DataFrame(
    data[:10],
    columns = ['単語1', '単語2', '人間', 'ベクトル']
)


from scipy.stats import spearmanr

def rank(x):
    args = np.argsort(-np.array(x))
    rank = np.empty_like(args)
    rank[args] = np.arange(len(x))
    return rank

human = [float(lst[2]) for lst in data]
w2v = [lst[3] for lst in data]
human_rank = rank(human)
w2v_rank = rank(w2v)
rho, p_value = spearmanr(human_rank, w2v_rank)

print('順位相関係数 :', rho)
print('p値 :', p_value)

import matplotlib.pyplot as plt

plt.scatter(human_rank, w2v_rank)
plt.show()



#################################################################################
# 67. k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

df670 = df01[df01[0].isin( [ ': capital-common-countries\n']) ]
df671 = df01[df01[0].isin( [ ': capital-world\n']) ]
df672 = df01[df01[0].isin( [  ': currency\n']) ]


countries = list(set( df670.iloc[:,2] ) | 
                 set( df670.iloc[:,4] ) |
                 set( df671.iloc[:,2] ) |
                 set( df671.iloc[:,4] ) |
                 set( df672.iloc[:,1] ) |
                 set( df672.iloc[:,3] ) 
                 )
print(len(countries))


country_vectors = [model[country] for country in countries]


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(country_vectors)


for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('クラス', i)
    print(', '.join([countries[k] for k in cluster]))





#################################################################################
# 68. Ward法によるクラスタリング
# 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
# さらに，クラスタリング結果をデンドログラムとして可視化せよ．

from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(16, 9), dpi=200)
Z = linkage(country_vectors, method='ward')
dendrogram(Z, labels = countries)
plt.show()



#################################################################################
# 69. t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．

from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit(country_vectors)

plt.figure(figsize=(15, 15), dpi=300)
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1])
for (x, y), name in zip(tsne.embedding_, countries):
    plt.annotate(name, (x, y))
plt.show()


"""
https://qiita.com/g-k/items/120f1cf85ff2ceae4aba

t-SNEは高次元データを2次元や3次元に落とし込むための次元削減アルゴリズムです。
次元削減といえば古典的なものとしてPCAやMDSがありますが、それら線形的な次元削減にはいくつかの問題点がありました。

異なるデータを低次元上でも遠くに保つことに焦点を当てたアルゴリズムのため、類似しているデータを低次元上でも近くに保つことには弱い
特に高次元上の非線形的なデータに対しては「類似しているデータを低次元上でも近くに保つこと」は不可能に近い
これらの問題点を解決するためにデータの局所的な構造(類似しているデータを低次元上でも近くに保つこと)の維持を目的とした非線形次元削減技術が色々と生み出されました。
t-SNEはその流れを汲んだアルゴリズムになります。
"""












