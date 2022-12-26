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
44. 係り受け木の可視化
与えられた文の係り受け木を有向グラフとして可視化せよ．
可視化には，Graphviz等を用いるとよい．

## https://qiita.com/vegetal/items/6d8f036ab5b7b37b3d74

"""

## setting_parameter ####

N=1  

#########################



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



def f03(nn_):
    bb=blocks[nn_]
    for j,x in enumerate(bb):
        dst01=x[1].dst
        z=bb[dst01]
        if dst01 != -1 :
            print(N,
                  x[0],
                  dst01,
                  ''.join([ xx.surface for xx in x[1].morphs if xx.pos!='記号' ]), 
                  ''.join([ zz.surface for zz in z[1].morphs if zz.pos!='記号' ]) 
                  ,sep='\t')

# f03(N)



def f04(nn_):
    bb=blocks[nn_]
    for j,x in enumerate(bb):
        dst01=x[1].dst
        z=bb[dst01]
        xxx=[ xx.pos for xx in x[1].morphs if xx.pos=='名詞']
        zzz=[ zz.pos for zz in z[1].morphs if zz.pos=='動詞']
        if dst01 != -1 and len(xxx)>0 and len(zzz)>0 :
            print(N,
                  x[0],
                  dst01,
                  ''.join([ xx.surface for xx in x[1].morphs if xx.pos!='記号' ]), 
                  ''.join([ zz.surface for zz in z[1].morphs if zz.pos!='記号' ]) 
                  ,sep='\t')


# f04(N)



def f05(nn_):
    print('文番号: ', nn_)
    from graphviz import Digraph
    graph = Digraph(format="png")

    bb=blocks[nn_]
    for j,x in enumerate(bb):
        dst01=x[1].dst
        z=bb[dst01]
        if dst01 != -1 :
            node1 = ''.join([ xx.surface for xx in x[1].morphs if xx.pos!='記号' ])
            node2 = ''.join([ zz.surface for zz in z[1].morphs if zz.pos!='記号' ]) 
            graph.edge(node1, node2)
            
    graph.render("image/output")
    graph.view()
            

f05(N)








