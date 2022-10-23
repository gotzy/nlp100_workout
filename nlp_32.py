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
32. 動詞の基本形
動詞の基本形をすべて抽出せよ．
"""


import pickle
with open('./04_dir/res_q30.pickle', 'rb') as f:
    ls_res = pickle.load(f)


ls_res_a=list()
for x in ls_res:
    ls_res_b=list()
    for y in x:
        if y['pos']=='動詞':
            ls_res_b.append(y['base'])
    ls_res_a.append(ls_res_b)
            
# ls_res_a is the result of q_32.


            
        




