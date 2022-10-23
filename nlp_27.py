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
27. 内部リンクの除去Permalink
26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ
（参考: マークアップ早見表）．
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

# q_27



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
                    z=re.sub( "\||'+(.*?)'+", '', y)
                    z=re.sub( "\[(.*?#\.*?\|.*?)\]\]", '\\1' ,z )
                    z=re.sub( "\[(.*?\|.*?)\]\]", '\\1' ,z )
                    z=re.sub( "\[(.*?)\]\]", '\\1' ,z )
                    z=re.sub( "\[(.*?)\]\]", '\\1' ,z )
                    if len(re.split('\s+=\s*', z ))==2:
                        ls_res.append( re.split('\s+=\s*', z ) )

                

from collections import OrderedDict
                
dt_res=OrderedDict()

for a,b in ls_res:
    dt_res[a]=b









