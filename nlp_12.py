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

12. 1列目をcol1.txtに，2列目をcol2.txtに保存
各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
確認にはcutコマンドを用いよ．
"""


# UNIX
'''

!cut -f 1  02_dir/popular-names.txt  >  02_dir/nlp100_12_unix_res_col01.txt

!cut -f 2  02_dir/popular-names.txt  >  02_dir/nlp100_12_unix_res_col02.txt

'''


# python_prg
with open('./02_dir/popular-names.txt') as f:
    res=f.readlines()

res1=[ x.split('\t')[0]  for i,x in enumerate(res)   ]
res2=[ x.split('\t')[1]  for i,x in enumerate(res)   ]
    
with open('./02_dir/nlp100_12_py_res_col01.txt', 'w') as f:
    for y in res1:
        print(y,file=f)
        
with open('./02_dir/nlp100_12_py_res_col02.txt', 'w') as f:
    for y in res2:
        print(y,file=f)
        





