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

10. 行数のカウント
行数をカウントせよ．確認にはwcコマンドを用いよ．
-> 2780
"""


# UNIX
'''
wc -l 02_dir/popular-names.txt
'''


# python_prg
with open('./02_dir/popular-names.txt') as f:
    res=f.readlines()

print(len(res))
    





