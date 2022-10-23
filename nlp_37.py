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
37. 「猫」と共起頻度の高い上位10語
「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．
"""



import pandas as pd
import pickle
with open('./04_dir/res_q30.pickle', 'rb') as f:
    ls_res = pickle.load(f)


# 共起窓の広さ：N個両隣までを共起とする。
# N=int(input())
N=1

ls_res_a=list()
for x in ls_res:
    for j,y in enumerate(x):
        if y['base']=='猫':

            for k in range(1,N+1):
                try:
                    ls_res_a.append(x[j-k]['base'])
                except:
                    pass
                try:
                    ls_res_a.append(x[j+k]['base'])
                except:
                    pass



df01=pd.DataFrame(ls_res_a,columns=['word'])
df01['cnt']=list(range(df01.shape[0]))

df02=df01.groupby('word',as_index=False)['cnt'].count()
df_res=df02.sort_values(by='cnt', ascending=False)
df_res2=df_res.iloc[:10,:]




import matplotlib.pyplot as plt
import japanize_matplotlib

# matplotlib version
plt.bar(df_res2['word'],df_res2['cnt'])
plt.show()

# pandas_plot_version
df_res2.index=list(df_res2['word'])
df_res2['cnt'].plot.bar()









