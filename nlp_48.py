#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch05.html

第5章: 係り受け解析
日本語Wikipediaの「人工知能」に関する記事からテキスト部分を抜き出したファイルがai.ja.zipに収録されている．
この文章をCaboChaやKNP等のツールを利用して係り受け解析を行い，
その結果をai.ja.txt.parsedというファイルに保存せよ．
このファイルを読み込み，以下の問に対応するプログラムを実装せよ．

---

https://nlp100.github.io/data/ai.ja.zip
をダウンロードし、以下のように格納した。
./05_dir/ai.ja.zip
この前提で進める。

以下のコマンド
unzip ai.ja.zip  # ai.ja.txt ができる
cat ai.ja.txt | cabocha -f1 > ai.ja.txt.parsed
で係り受け解析し、以下のように保存
./05_dir/ai.ja.txt.parsed

---
48. 名詞から根へのパスの抽出Permalink
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． 
ただし，構文木上のパスは以下の仕様を満たすものとする．

各文節は（表層形の）形態素列で表現する
パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．
---
ジョンマッカーシーは -> 作り出した
AIに関する -> 最初の -> 会議で -> 作り出した
最初の -> 会議で -> 作り出した
会議で -> 作り出した
人工知能という -> 用語を -> 作り出した
用語を -> 作り出した
--

"""



class Morph:
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']


class Chunk:
    def __init__(self, dst):
        self.morphs = []
        self.dst = dst  
        self.srcs = []



def f02(block_):
    res=list()
    idx_pairs=list()
    
    
    for line in block_.split('* ')[1:]:
        ls01=line.split('\n')[:-1]
        for ii, line2 in enumerate(ls01):
            if ii==0:
                dst=int(line2.split(' ')[1].rstrip('D'))
                src=int(line2.split(' ')[0])
                c=Chunk(dst)
                idx_pairs.append([ src , dst ])
            else:
                (surface, attr) = line2.split('\t')
                attr = attr.split(',')
                lineDict = {
                    'surface': surface,
                    'base': attr[6],
                    'pos': attr[0],
                    'pos1': attr[1]
                }
                c.morphs.append(Morph(lineDict))
                
            if ii==len(ls01)-1:
                res.append([src,c])
                
    
    for r1,r2 in res:
        for ip1,ip2 in idx_pairs:
            if r1==ip2:
                r2.srcs.append(ip1)
    
    return res
        
    
    
filename = '05_dir/ai.ja.txt.parsed'
with open(filename, mode='rt', encoding='utf-8') as f:
    blocks = [ x for x in f.read().split('EOS\n') if x!='' ]
blocks = [f02(block) for block in blocks]



# filename = '05_dir/input_f2'
# with open(filename, mode='rt', encoding='utf-8') as f:
#     blocks = [ x for x in f.read().split('EOS\n') if x!='' ]
# blocks = [f02(block) for block in blocks]



def f08(n_):
    chunks=blocks[n_]
    
    from collections import OrderedDict
    od = OrderedDict()
    
    ## lp01
    for i,x in enumerate(chunks):
        od[i]=x[1].dst
    
    ## lp02
    res1=list()
    for i,x in enumerate(chunks):
        a=i
        res2=[i]
        cond=0
        ls01 = [ y.pos for  y in chunks[a][1].morphs if y.pos=='名詞' ]
        if (len(ls01)==0) and (od[a]!=-1):
                cond=1
        while a!=-1:
            a=od[a]
            res2.append(a)
        if cond!=1:    
            res1.append(res2[:-1])
    res1=res1[:-1]
    
    ## lp03
    route=[ ' -> '.join([ ''.join([ z.surface for z in chunks[y][1].morphs if z.pos!='記号'  ])  for y in x ])   for x in res1]
        
    for r in route:
        print(r)
    



f08(1)



