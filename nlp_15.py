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

15. 末尾のN行を出力
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．
確認にはtailコマンドを用いよ．
"""


# UNIX
'''

(最後から5行目までの場合)
tail -n 5 02_dir/popular-names.txt

'''


# python_prg

N = int(input('please input number: '))

with open('./02_dir/popular-names.txt') as f:
    res=[ x.rstrip('\n') for x in f.readlines() ]
  
    
for i,y in enumerate(res):
    if i>=len(res)-N:
        print(y)
    else:
        pass
        





