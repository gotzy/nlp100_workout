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

11. タブをスペースに置換
タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．
"""


# UNIX
'''
macos(brew install gnu-sed)

gsed -r 's/\t/ /g'  02_dir/popular-names.txt  >  02_dir/nlp100_11_unix_res

'''


# python_prg
with open('./02_dir/popular-names.txt') as f:
    res=f.read()


res=res.replace('\t',' ')
    
with open('./02_dir/nlp100_11_py_res', 'w') as f:
    f.write(res)





