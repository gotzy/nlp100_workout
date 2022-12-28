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
47. 機能動詞構文のマイニング
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
46のプログラムを以下の仕様を満たすように改変せよ．

「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，以下の出力が得られるはずである．
--
学習を行う	に を	元に 経験を
--

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



from collections import OrderedDict

with open('05_dir/output_47','w') as f:
    pass
    
def f07(nn_):
    od = OrderedDict()
    
    bb=blocks[nn_]
    
    for j,x in enumerate(bb): # sentence -> chunk
        xxx=[ xx.pos for xx in x[1].morphs if xx.pos=='動詞']
        aaa=x[1].srcs
        if  len(xxx)>0 and len(aaa)>0 :
            vb=''.join([ xx.base for xx in x[1].morphs if xx.pos=='動詞' ][:1]) 
            for p in aaa:
                pt0=''
                z=bb[p] # sentence -> chunk
                
                # chunk -> morph
                if z[1].morphs[-1].surface=='を' and z[1].morphs[-1].pos=='助詞' and  z[1].morphs[-2].pos1=='サ変接続' and z[1].morphs[-2].pos=='名詞'  :
                    pt0 = z[1].morphs[-2].surface + z[1].morphs[-1].surface
                    
                else:
                    if z[1].morphs[-1].pos=='助詞':
                        pt = z[1].morphs[-1].surface
                        pt2 = ''.join([zz.surface for zz in z[1].morphs ])
                    
                    
                if pt0!='':
                    try:
                        od[pt0+vb] = od.get(pt0+vb,[]) + [[pt,pt2]]
                    except:
                        pass
        
    
    with open('05_dir/output_47','a') as f:
        for x in od:
            y=' '.join([ a for a,b in sorted(od[x]) ] + [ b for a,b in sorted(od[x]) ])
            print(x , y , sep='\t' , file=f )            
            
            
for k in range(len(blocks)):
    f07(k)






