#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch03.html

第3章: 正規表現

Wikipediaの記事を以下のフォーマットで書き出したファイルjawiki-country.json.gzがある．

1行に1記事の情報がJSON形式で格納される
各行には記事名が”title”キーに，記事本文が”text”キーの辞書オブジェクトに格納され，
そのオブジェクトがJSON形式で書き出される
ファイル全体はgzipで圧縮される
以下の処理を行うプログラムを作成せよ．
---

https://nlp100.github.io/data/jawiki-country.json.gz
をダウンロードし、以下のように格納した。
./03_dir/jawiki-country.json.gz
この前提で進める。

---
22. カテゴリ名の抽出
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．
"""

# q_20
import json
import gzip

articles=list()
with gzip.open('./03_dir/jawiki-country.json.gz', 'r') as f:
    for line in f:
        obj = json.loads(line)
        articles.append(obj)

####

import re

# q_22
for i,x in enumerate(articles):
    if x['title']=='イギリス':
        data=x['text'].split('\n')
        for j,y in enumerate(data):
            if y.startswith('[[Category'):
                # print( x['title'] , y )
                
                res = re.sub( '\[\[Category:|\]+|\|\*' , '' , y )
    
                print(res)












