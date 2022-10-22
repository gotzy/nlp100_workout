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
19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．
確認にはcut, uniq, sortコマンドを用いよ．
"""


# UNIX
'''

cut -f1 02_dir/popular-names.txt | sort -k1 | uniq -c | sort -k1r 

'''


# python_prg
with open('./02_dir/popular-names.txt') as f:
    res1=[x.rstrip('\n') for x in f.readlines()]


res2=[ x.rstrip('\n').split('\t')[0] for x in res1 ]

res3=sorted(res2,reverse=True)


ls_res=list()
cnt=1
ref=''
for j,z in enumerate(res3):
    
    
    if ref!=z:
        ls_res.append([cnt, ref])
        cnt=1
    else:
        cnt+=1
        
    ref=z

ls_res2=sorted(ls_res[1:], reverse=True)

for a,b in ls_res2:
    print(a,b)












