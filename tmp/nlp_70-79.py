#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch08.html


第8章: ニューラルネット
第6章で取り組んだニュース記事のカテゴリ分類を題材として，ニューラルネットワークでカテゴリ分類モデルを実装する．
なお，この章ではPyTorch, TensorFlow, Chainerなどの機械学習プラットフォームを活用せよ．

--
以下を参考。
https://www.takapy.work/entry/2021/07/03/125911

"""

#################################################################################
# 70. 単語ベクトルの和による特徴量

import numpy as np
import pandas as pd
import gensim
import datetime

t1=datetime.datetime.now()


with open('06_dir/train.txt') as f:
    tr_lines=[ x.rstrip('\n').split('\t') for x in f.readlines() ]
    
with open('06_dir/valid.txt') as f:
    vl_lines=[ x.rstrip('\n').split('\t') for x in f.readlines() ]
    
with open('06_dir/test.txt') as f:
    te_lines=[ x.rstrip('\n').split('\t') for x in f.readlines() ]
    
tr_lines=pd.DataFrame(tr_lines)
vl_lines=pd.DataFrame(vl_lines)
te_lines=pd.DataFrame(te_lines)



model = gensim.models.KeyedVectors.load_word2vec_format('07_dir/GoogleNews-vectors-negative300.bin', binary=True)


def f70(data_):
    wd_list = data_.split()
    ls01=list()
    for wd in wd_list:
        try:
            v=model[wd]
            ls01.append(v)
        except:
            pass
        
    try:
        res=list(sum(ls01) / len(ls01))
    except:
        res=[0]*300
        
    return res

tr_lines[1] = [ f70(x) for x in tr_lines[1] ]  
vl_lines[1] = [ f70(x) for x in vl_lines[1] ]  
te_lines[1] = [ f70(x) for x in te_lines[1] ]  


tr_lines[0] = [ 0 if x=='b' else 1 if x=='t' else 2 if x=='e' else 3  for x in tr_lines[0] ]  
vl_lines[0] = [ 0 if x=='b' else 1 if x=='t' else 2 if x=='e' else 3  for x in vl_lines[0] ]  
te_lines[0] = [ 0 if x=='b' else 1 if x=='t' else 2 if x=='e' else 3  for x in te_lines[0] ]  



x_train = np.array(list(tr_lines[1]))
x_valid = np.array(list(vl_lines[1]))
x_test = np.array(list(te_lines[1]))

y_train = list(tr_lines[0])
y_valid = list(vl_lines[0])
y_test = list(te_lines[0])



res70=[ tr_lines, vl_lines, te_lines ]

# import pickle

# with open('08_dir/res70.pickle', 'wb') as f:
#     pickle.dump(res70, f)

# with open('08_dir/res70.pickle', 'rb') as f:
#     tr_lines, vl_lines, te_lines = pickle.load(f)



t2=datetime.datetime.now()
print(t2-t1)


#################################################################################
# 71. 単層ニューラルネットワークによる予測

import numpy as np

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u



import pandas as pd
import tensorflow as tf


class SimpleNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        output_layer = self.output(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model



model = SimpleNet(x_train.shape[1], 4).build()

print(model(x_train[:1]))
print(model(x_train[:4]))



#################################################################################
# 72. 損失と勾配の計算


import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class SimpleNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        output_layer = self.output(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model




# モデル構築
model = SimpleNet(x_train.shape[1], len( set(y_train) )).build()
preds = model(x_train[:4])

# 目的変数をone-hotに変換
y_true = to_categorical(y_train)
y_true = y_true[:4]

# 計算
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, preds.numpy()).numpy())

    



#################################################################################
# 73. 確率的勾配降下法による学習


import pandas as pd
import tensorflow as tf


class SimpleNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        output_layer = self.output(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model


# モデル構築
model = SimpleNet(x_train.shape[1], len(set(y_train))).build()
opt = tf.optimizers.SGD()
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
)

# 学習
tf.keras.backend.clear_session()

model.fit(
    x_train,
    np.array(y_train),
    epochs=50,
    batch_size=32,
    verbose=1
)

# モデルの保存
model.save("08_dir/tf_model")



#################################################################################
# 74. 正解率の計測


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


class SimpleNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        output_layer = self.output(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model



# モデルのロード
model = tf.keras.models.load_model("08_dir/tf_model")

# 推論
y_train_preds = model.predict(x_train, verbose=1)
y_valid_preds = model.predict(x_valid, verbose=1)

# 一番確率の高いクラスを取得
y_train_preds = np.argmax(y_train_preds, 1)
y_valid_preds = np.argmax(y_valid_preds, 1)

# 正解率を出力
print(f'Train Accuracy: {accuracy_score(y_train, y_train_preds)}')
print(f'Valid Accuracy: {accuracy_score(y_valid, y_valid_preds)}')


#################################################################################
"""
75. 損失と正解率のプロット
問題73のコードを改変し，各エポックのパラメータ更新が完了するたびに，訓練データでの損失，正解率，検証データでの損失，正解率をグラフにプロットし，学習の進捗状況を確認できるようにせよ．

76. チェックポイント
問題75のコードを改変し，各エポックのパラメータ更新が完了するたびに，チェックポイント（学習途中のパラメータ（重み行列など）の値や最適化アルゴリズムの内部状態）をファイルに書き出せ．

77. ミニバッチ化
問題76のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ（ミニバッチ化）．Bの値を1,2,4,8,…と変化させながら，1エポックの学習に要する時間を比較せよ．
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class SimpleNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        output_layer = self.output(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model



# モデル構築
model = SimpleNet(x_train.shape[1], len(set(y_train))).build()
opt = tf.optimizers.SGD()
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# チェックポイント
checkpoint_path = '08_dir/ck_tf_model'
cb_checkpt = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
# 学習
tf.keras.backend.clear_session()
history = model.fit(
    x_train,
    np.array(y_train),
    epochs=100,
    batch_size=32,
    callbacks=[cb_checkpt],
    verbose=1
)

# 学習曲線の保存
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.savefig("08_dir/learning_curves.png")





#################################################################################

"""
79. 多層ニューラルネットワーク

問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ．
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


class MLPNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.hidden1 = tf.keras.layers.Dense(128, activation='relu', name='hidden1')
        self.hidden2 = tf.keras.layers.Dense(32, activation='relu', name='hidden2')
        self.dropout = tf.keras.layers.Dropout(0.2, name='dropout')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        hidden1 = self.hidden1(input_layer)
        dropout1 = self.dropout(hidden1)
        hidden2 = self.hidden2(dropout1)
        dropout2 = self.dropout(hidden2)
        output_layer = self.output(dropout2)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model




# モデル構築
model = MLPNet(x_train.shape[1], len(set(y_train))).build()
opt = tf.optimizers.SGD()
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# チェックポイント
checkpoint_path = '08_dir/ck_tf_model'
cb_checkpt = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
# 学習
tf.keras.backend.clear_session()
history = model.fit(
    x_train,
    np.array(y_train),
    epochs=100,
    batch_size=32,
    callbacks=[cb_checkpt],
    verbose=1
)

# 推論
y_train_preds = model.predict(x_train, verbose=1)
y_valid_preds = model.predict(x_valid, verbose=1)

# 一番確率の高いクラスを取得
y_train_preds = np.argmax(y_train_preds, 1)
y_valid_preds = np.argmax(y_valid_preds, 1)

# 正解率を出力
print(f'Train Accuracy: {accuracy_score(y_train, y_train_preds)}')
print(f'Valid Accuracy: {accuracy_score(y_valid, y_valid_preds)}')

# 学習曲線の保存
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.savefig("learning_curves.png")










