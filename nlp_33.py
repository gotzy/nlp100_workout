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
33. 「AのB」
2つの名詞が「の」で連結されている名詞句を抽出せよ．
"""


import pickle
with open('./04_dir/res_q30.pickle', 'rb') as f:
    ls_res = pickle.load(f)


ls_res_a=list()
for x in ls_res:
    ls_res_b=list()
    for j,y in enumerate(x):
        try:
            if x[j-1]['pos']=='名詞' and x[j+1]['pos']=='名詞' and x[j]['surface']=='の':
                ls_res_b.append( x[j-1]['surface'] + x[j]['surface'] + x[j+1]['surface'] )
        except:
            pass

    ls_res_a.append(ls_res_b)

            
# ls_res_a is the result of q_33.


for x in ls_res_a:
    if len(x)!=0:
        print(x)

        




