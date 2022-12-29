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

--

コピペして勉強した。

https://ds-blog.tbtech.co.jp/entry/2020/08/28/%E2%96%B2%E5%BF%83%E3%81%8F%E3%81%98%E3%81%91%E3%81%9A%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86100%E6%9C%AC%E3%83%8E%E3%83%83%E3%82%AF%EF%BC%9D%EF%BC%9D50%EF%BD%9E54%EF%BC%9D%EF%BC%9D

https://ds-blog.tbtech.co.jp/entry/2020/09/01/%E2%96%B2%E5%BF%83%E3%81%8F%E3%81%98%E3%81%91%E3%81%9A%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86100%E6%9C%AC%E3%83%8E%E3%83%83%E3%82%AF%EF%BC%9D%EF%BC%9D55%EF%BD%9E59%EF%BC%9D%EF%BC%9D



"""



#################################################
# 51. 特徴量抽出
# 学習データ，検証データ，評価データから特徴量を抽出し，
# それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
# なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
# 記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

import pandas as pd
import random
import string
import re
import numpy as np


train_df = pd.read_csv('06_dir/train.txt', sep='\t', header=None)
val_df = pd.read_csv('06_dir/valid.txt', sep='\t', header=None)
test_df = pd.read_csv('06_dir/test.txt', sep='\t', header=None)


df = pd.concat([train_df, val_df, test_df], axis=0)

df.reset_index(drop=True, inplace=True)



#######################
#前処理の関数
def preprocessing(text):
  #記号(string.punctuation)をスペース(記号があった数分)に置換
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  text = text.translate(table)
  #数字を全て0に置換
  text = re.sub(r'[0-9]+', "0", text)
  return text
#前処理の実行
df[1] = df[1].map(lambda x: preprocessing(x))


#######################

#再度分割(train_valとtest)
train_val = df[:len(train_df)+len(val_df)]
test = df[len(train_df)+len(val_df):]
#CountVectorizerクラスを利用したBoW形式の特徴抽出(train_valのみ)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
train_val_cv = cv.fit_transform(train_val[1])
test_cv = cv.transform(test[1])
#データフレームに変換
train_val_ = pd.DataFrame(train_val_cv.toarray(), columns=cv.get_feature_names())
x_test = pd.DataFrame(test_cv.toarray(), columns=cv.get_feature_names())


#######################

#再度データを分割
x_train = train_val_[:len(train_df)]
x_val = train_val_[len(train_df):]


import datetime

t1=datetime.datetime.now()
print(t1)

#保存
x_train.to_csv('06_dir/train.feature.txt', sep='\t', index=False)
x_val.to_csv('06_dir/valid.feature.txt', sep='\t', index=False)
x_test.to_csv('06_dir/test.feature.txt', sep='\t', index=False)


t2=datetime.datetime.now()
print(t2)

print(t2-t1)



# #######################################################################################
# 52. 学習
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

#ロジスティック回帰による学習
y_train=train_df[0]
y_val=val_df[0]
y_test=test_df[0]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200, random_state=0)
model.fit(x_train, y_train)


# ##出力
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=200,
#                    multi_class='auto', n_jobs=None, penalty='l2',
#                    random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
#                    warm_start=False)


# ####################################################################################
# 53. 予測
# 52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

#予測用の関数
def predict(model, data):
    return [np.max(model.predict_proba(data), axis=1), model.predict(data)]
#関数の実行
train_predict = predict(model, x_train)
test_predict = predict(model, x_test)
print(train_predict)
print(test_predict)




####################################################################################
# 54. 正解率の計測
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, train_predict[1])
test_accuracy = accuracy_score(y_test, test_predict[1])
print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')



####################################################################################
# 55. 混同行列の作成
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#学習データの混同行列
train_cm = confusion_matrix(y_train, train_predict[1])
print(train_cm)
sns.heatmap(train_cm, annot=True, cmap='Blues')
plt.show()

#評価データの混同行列
test_cm = confusion_matrix(y_test, test_predict[1])
print(test_cm)
sns.heatmap(test_cm, annot=True, cmap='Blues')
plt.show()



####################################################################################
# 56. 適合率，再現率，F1スコアの計測
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
# カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

#適合率
from sklearn.metrics import precision_score
pre = precision_score(y_test, test_predict[1], average=None, labels=['b','t','e','m'])
print (pre)
micro_average = precision_score(y_test, test_predict[1], average='micro')
macro_average = precision_score(y_test, test_predict[1], average='macro')
print('ミクロ平均：' + str(micro_average))
print('マクロ平均：' + str(macro_average))

#結合
pre = np.append(pre, [micro_average, macro_average])
print(pre)


#再現率
from sklearn.metrics import recall_score
rec = recall_score(y_test, test_predict[1], average=None, labels=['b','t','e','m'])
micro_average = recall_score(y_test, test_predict[1], average='micro')
macro_average = recall_score(y_test, test_predict[1], average='macro')
rec = np.append(rec, [micro_average, macro_average])

#Fスコア
from sklearn.metrics import f1_score
f1 = f1_score(y_test, test_predict[1], average=None, labels=['b','t','e','m'])
micro_average = f1_score(y_test, test_predict[1], average='micro')
macro_average = f1_score(y_test, test_predict[1], average='macro')
f1 = np.append(f1, [micro_average, macro_average])

#データフレーム化
scores = pd.DataFrame({'適合率':pre, '再現率':rec, 'F1スコア':f1},
                      index=['b','e','t','m','micro','macro'])




####################################################################################
# 57. 特徴量の重みの確認
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

#カラム名
features = x_train.columns.values
#重みの高い特徴量トップ10
for c,coef in zip(model.classes_, model.coef_):
  idx = np.argsort(coef)
  print(c,features[idx][-10:][::-1])


#重みの低い特徴量トップ10
for c,coef in zip(model.classes_, model.coef_):
  idx = np.argsort(coef)
  print(c,features[idx][:10])




####################################################################################
# 58. 正則化パラメータの変更
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，
# 学習時の過学習（overfitting）の度合いを制御できる．
# 異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
# 実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

result = []
C = np.logspace(-5, 4, 10, base=10)
for c in C:
    #モデルの学習
    model = LogisticRegression(random_state=0, max_iter=5000, C=c)
    model.fit(x_train, y_train)
    #それぞれの予測値
    train_predict = predict(model, x_train)
    val_predict = predict(model, x_val)
    test_predict = predict(model, x_test)
    #正解率の計算
    train_acc = accuracy_score(y_train, train_predict[1])
    val_acc = accuracy_score(y_val, val_predict[1])
    test_acc = accuracy_score(y_test, test_predict[1])
    #resultに格納
    result.append([c, train_acc, val_acc, test_acc])
result = np.array(result).T
print(result)


#可視化
plt.plot(result[0], result[1], label='train')
plt.plot(result[0], result[2], label='val')
plt.plot(result[0], result[3], label='test')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.xscale('log')
plt.xlabel('C')
plt.legend()
plt.show()



####################################################################################
# 59. ハイパーパラメータの探索
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
# 検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
# また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

import itertools
def calc_scores(C,class_weight):
    #モデルの宣言
    model = LogisticRegression(random_state=0, max_iter=10000, C=C, class_weight=class_weight)
    #モデルの学習
    model.fit(x_train, y_train)
    #モデルの検証
    y_train_pred = model.predict(x_train)
    y_valid_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)
    #スコア
    scores = []
    scores.append(accuracy_score(y_train, y_train_pred))
    scores.append(accuracy_score(y_val, y_valid_pred))
    scores.append(accuracy_score(y_test, y_test_pred))
    return scores


# Cとclass_weightの総当たりの組み合わせを試します
C = np.logspace(-5, 4, 10, base=10)
class_weight = [None, 'balanced']
best_parameter = None
best_scores = None
max_valid_score = 0


#itertools.product()で全ての組み合わせを作成
for c, w in itertools.product(C, class_weight):
    # ハイパーパラメータの組み合わせの表示
    print(c, w)
    #ハイパーパラメータの組み合わせで関数の実行
    scores = calc_scores(c, w)
    #前のスコアより高ければ結果を更新
    if scores[1] > max_valid_score:
      max_valid_score = scores[1]
      best_parameter = [c, w]
      best_scores = scores
    
    
#最適なハイパーパラメータの組み合わせとスコアの表示
print ('C: ', best_parameter[0], 'solver: ', 'class_weight: ', best_parameter[1])
print ('best scores: ', best_scores)
print ('test accuracy: ', best_scores[2])


















