#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch04.html

第4章: 形態素解析


夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，
その結果をneko.txt.mecabというファイルに保存せよ．
このファイルを用いて，以下の問に対応するプログラムを実装せよ．

なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．

---

https://nlp100.github.io/data/neko.txt
をダウンロードし、以下のように格納した。
./04_dir/neko.txt
この前提で進める。

mecabを使った以下のコマンド
cat neko.txt | mecab > neko.txt.mecab
で形態素解析し、以下のように保存
./04_dir/neko.txt.mecab

---
39. Zipfの法則
単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
"""

import pandas as pd
import pickle
with open('./04_dir/res_q30.pickle', 'rb') as f:
    ls_res = pickle.load(f)


ls_res_a=list()
for x in ls_res:
    ls_res_b=list()
    for j,y in enumerate(x):
        ls_res_a.append(y['base'])


df01=pd.DataFrame(ls_res_a,columns=['word'])
df01['cnt']=list(range(df01.shape[0]))

df02=df01.groupby('word',as_index=False)['cnt'].count()
df_res=df02.sort_values(by='cnt', ascending=False)
df_res['rank']=list(range(1, df_res.shape[0]+1))



import matplotlib.pyplot as plt
plt.plot(df_res['rank'] , df_res['cnt'])
plt.yscale('log')
plt.xscale('log')
plt.show()











