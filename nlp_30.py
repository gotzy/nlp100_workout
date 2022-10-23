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
30. 形態素解析結果の読み込み
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）を
キーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．

第4章の残りの問題では，ここで作ったプログラムを活用せよ．

"""


# from collections import OrderedDict
# surface: 0
# base: 7
# pos: 1
# pos1: 2


# q30
with open('./04_dir/neko.txt.mecab') as f:
    ls=[ x.rstrip('\n') for i,x in enumerate(f.readlines()) if x!='\n' ]
    
    
ls_res=list() # <- res_q30
ls_res_2=list()
for i,x in enumerate(ls):
    # dt_res=OrderedDict()
    dt_res=dict()
    y=x.rstrip('\n')
    if x=='EOS':
        ls_res.append(ls_res_2)
        ls_res_2=list()        
    else:
        z=y.split('\t')
        a=[z[0]]
        b=z[1].split(',')
        res=a+b
        dt_res['surface'] = res[0]
        dt_res['base'] = res[7]
        dt_res['pos'] = res[1]
        dt_res['pos1'] = res[2]
        ls_res_2.append(dt_res)
            
        
import pickle
with open('./04_dir/res_q30.pickle', 'wb') as f:
    pickle.dump(ls_res, f)

        

# import pickle
# with open('./04_dir/res_q30.pickle', 'rb') as f:
#     ls_res = pickle.load(f)




            
        




