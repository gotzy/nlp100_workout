#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch02.html

popular-names.txtは，アメリカで生まれた赤ちゃんの「名前」「性別」「人数」「年」をタブ区切り形式で格納したファイルである．
以下の処理を行うプログラムを作成し，popular-names.txtを入力ファイルとして実行せよ．
さらに，同様の処理をUNIXコマンドでも実行し，プログラムの実行結果を確認せよ．
---

https://nlp100.github.io/data/popular-names.txt
をダウンロードし、以下のように格納した。
./02_dir/popular-names.txt
この前提で進める。

---

https://analytics-note.xyz/programming/subprocess-pipeline/

---

16. ファイルをN分割する
自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．
同様の処理をsplitコマンドで実現せよ．
"""


# UNIX
'''
(macos: brew install coreutils)

(3分割で、最大927行ずつ各ファイルに格納する場合)
gsplit -l 927 02_dir/popular-names.txt  02_dir/

'''


# python_prg
N = int(input('please input number: '))

with open('./02_dir/popular-names.txt') as f:
    res=[ x.rstrip('\n') for x in f.readlines() ]
    
quotient = len(res) // N
remainder = len(res) % N

ls01=[ quotient+1 if i<=remainder-1 else quotient  for i in range(N) ]

ls02 = [ sum(ls01[:i]) for i,x in enumerate(ls01)]
ls03 = [ sum(ls01[:i+1]) for i,x in enumerate(ls01)]


i=0
for a,b in zip(ls02,ls03):
    i+=1
    with open('./02_dir/nlp100_16_res_{}.txt'.format(i), 'w') as f:
        for j,z in enumerate(res):
            if a<=j and j<b: 
                print(z , file=f)
        






