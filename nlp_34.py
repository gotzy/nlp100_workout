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
34. 名詞の連接
名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．
"""


import pickle
with open('./04_dir/res_q30.pickle', 'rb') as f:
    ls_res = pickle.load(f)


ls_res_a=list()
for x in ls_res:
    ls_res_b=list()
    for j,y in enumerate(x):
        if (j==0 or x[j-1]['pos']!='名詞') and x[j]['pos']=='名詞':
            ls_res_c=list()
            for k in range(j,len(x)):
                if x[k]['pos']=='名詞':
                    ls_res_c.append(x[k]['surface'])
                else:
                    break
            if len(ls_res_c)>=2:
                ls_res_b.append(''.join(ls_res_c))

    ls_res_a.append(ls_res_b)

            
# ls_res_a is the result of q_34.


for x in ls_res_a:
    if len(x)!=0:
        print(x)

        




