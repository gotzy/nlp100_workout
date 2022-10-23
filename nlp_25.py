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
25. テンプレートの抽出
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．
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

# q_25

ls_res=list()
for i,x in enumerate(articles):
    if x['title']=='イギリス':
        cond=0
        for y in x['text'].split('\n'):
            if y.startswith('{{基礎情報 '):
                cond=1
            if y.startswith('}}'):
                break
            
            if cond==1:
                if y.startswith('{{基礎情報 ') or y.startswith('*'):
                    pass
                else:
                    if len(re.split('\s+=\s*', y.replace('|','') ))==2:
                        ls_res.append( re.split('\s+=\s*', y.replace('|','') ) )



from collections import OrderedDict
                
dt_res=OrderedDict()

for a,b in ls_res:
    dt_res[a]=b

            









