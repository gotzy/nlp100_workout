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
29. 国旗画像のURLを取得するPermalink
テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）
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

# q_29



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
                    z=re.sub( "\||'+(.*?)'+", '\\1', y)
                    z=re.sub( "\[\[(.*?#\.*?\|.*?)\]\]", '\\1' ,z )
                    z=re.sub( "\[\[(.*?\|.*?)\]\]", '\\1' ,z )
                    z=re.sub( "\[\[(.*?)\]\]", '\\1' ,z )
                    z=re.sub( "\{\{(.*?)\}\}", '\\1' ,z )
                    
                    z=re.sub( "<ref (.*?)/>|<ref (.*?)(|/)>|<ref>(.*?)</ref>", '\\1' ,z )
                    
                    z=re.sub("<ref(|.*?)>|<br />",'',z)
                    
                    if len(re.split('\s+=\s*', z ))==2:
                        ls_res.append( re.split('\s+=\s*', z ) )

                

from collections import OrderedDict
                
dt_res=OrderedDict()

for a,b in ls_res:
    dt_res[a]=b


print('https://en.wikipedia.org/wiki/File:{0}'.format(dt_res['国旗画像']))


# import json
# import re
# from urllib import request, parse

# # リクエスト生成
# url = 'https://www.mediawiki.org/w/api.php?' \
#     + 'action=query' \
#     + '&titles=File:' + parse.quote(dt_res['国旗画像']) \
#     + '&format=json' \
#     + '&prop=imageinfo' \
#     + '&iiprop=url'

# # MediaWikiのサービスへリクエスト送信
# connection = request.urlopen(request.Request(url))

# # jsonとして受信
# response = json.loads(connection.read().decode())

# print(response['query']['pages']['-1']['imageinfo'][0]['url'])






