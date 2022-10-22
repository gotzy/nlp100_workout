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

13. col1.txtとcol2.txtをマージ
12で作ったcol1.txtとcol2.txtを結合し，
元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
確認にはpasteコマンドを用いよ．
"""


# UNIX
'''

paste ./02_dir/nlp100_12_py_res_col01.txt ./02_dir/nlp100_12_py_res_col02.txt > 02_dir/nlp100_13_unix_res

'''


# python_prg
with open('./02_dir/nlp100_12_py_res_col01.txt') as f:
    res1=[ x.rstrip('\n') for x in f.readlines()]
    
with open('./02_dir/nlp100_12_py_res_col02.txt') as f:
    res2=[ x.rstrip('\n') for x in f.readlines()]


    
with open('./02_dir/nlp100_13_py_res', 'w') as f:
    for x,y in zip(res1,res2):
        print(x, y, sep='\t', file=f)





