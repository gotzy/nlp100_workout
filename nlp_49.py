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
49. 名詞間の係り受けパスの抽出
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

* 問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
* 文節iとjに含まれる名詞句はそれぞれ，XとYに置換する

また，係り受けパスの形状は，以下の2通りが考えられる．
* 文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
* 上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 
    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．
---
Xは | Yに関する -> 最初の -> 会議で | 作り出した
Xは | Yの -> 会議で | 作り出した
Xは | Yで | 作り出した
Xは | Yという -> 用語を | 作り出した
Xは | Yを | 作り出した
Xに関する -> Yの
Xに関する -> 最初の -> Yで
Xに関する -> 最初の -> 会議で | Yという -> 用語を | 作り出した
Xに関する -> 最初の -> 会議で | Yを | 作り出した
Xの -> Yで
Xの -> 会議で | Yという -> 用語を | 作り出した
Xの -> 会議で | Yを | 作り出した
Xで | Yという -> 用語を | 作り出した
Xで | Yを | 作り出した
Xという -> Yを
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



def f09(n_):
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
    
    od2 = OrderedDict()
    for x in res1:
        od2[x[0]] = x
    
    
    res11=list()
    for i,x in enumerate(chunks):
        ls01 = [ y.pos for  y in chunks[i][1].morphs if y.pos=='名詞' ]
        if len(ls01) > 0:
            res11.append(i)
    
    res12=list()
    for i in res11:
        for j in res11:
            if i<j:
                res12.append([i,j])
    
    import re
    
    for i_,j_ in res12:
        
        ls_i=od2[i_]
        ls_j=od2[j_]
        
        res21=list()
        tmp_revi=ls_i[::-1]
        tmp_revj=ls_j[::-1]
        
        for id01, el01 in enumerate(ls_j[::-1]):
            if el01 == ls_i[::-1][id01]:
                res21.append(el01)
                tmp_revi = tmp_revi[1:]
                tmp_revj = tmp_revj[1:]
            else:
                break
        
        res31=res21[::-1]
        res32=tmp_revi[::-1]
        res33=tmp_revj[::-1]
        
        if len(res33)==0:
            route_xy=' -> '.join([ ''.join([ 'X' if iy==0 and z.pos=='名詞' else  
                                           'Y' if y==res31[0] and z.pos=='名詞'  else  
                                           z.surface for iz,z in enumerate(chunks[y][1].morphs) if z.pos!='記号'  ]) 
                                 for iy,y in enumerate(res32+res31[:1]) ])
    
            route_xy=re.sub('X+','X',route_xy)
            route_xy=re.sub('Y+','Y',route_xy)
            
            print(route_xy)
             
        else:
            
            route_x=' -> '.join([ ''.join([ 'X' if iy==0 and z.pos=='名詞' else  
                                           z.surface for iz,z in enumerate(chunks[y][1].morphs) if z.pos!='記号'  ]) 
                                 for iy,y in enumerate(res32) ])   
        
            route_y=' -> '.join([ ''.join([ 'Y' if iy==0 and z.pos=='名詞' else  
                                           z.surface for iz,z in enumerate(chunks[y][1].morphs) if z.pos!='記号'  ]) 
                                 for iy,y in enumerate(res33) ])   
        
            route_r=' -> '.join([ ''.join([ z.surface for iz,z in enumerate(chunks[y][1].morphs) if z.pos!='記号'  ]) 
                                 for y in res31 ])   
        
            route_x=re.sub('X+','X',route_x)
            route_y=re.sub('Y+','Y',route_y)
        
            print( '{0} | {1} | {2}'.format(route_x, route_y, route_r) )
        



f09(1)





