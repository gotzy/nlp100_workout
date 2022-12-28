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
46. 動詞の格フレーム情報の抽出
45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ．

項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
この文は「作り出す」という１つの動詞を含み，
「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると解析された場合は，
次のような出力になるはずである．
--
作り出す	で は を	会議で ジョンマッカーシーは 用語を
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

with open('05_dir/output_46','w') as f:
    pass
    
def f07(nn_):
    od = OrderedDict()
    
    bb=blocks[nn_]
    for j,x in enumerate(bb):
        # dst01=x[1].dst
        # z=bb[dst01]
        xxx=[ xx.pos for xx in x[1].morphs if xx.pos=='動詞']
        aaa=x[1].srcs
        if  len(xxx)>0 and len(aaa)>0 :
            vb=''.join([ xx.base for xx in x[1].morphs if xx.pos=='動詞' ][:1]) 
            for p in aaa:
                z=bb[p]
                ppt=[ zz.surface for zz in z[1].morphs if zz.pos=='助詞' ][-1:]
                ppt2=[ zz.surface for zz in z[1].morphs ]
                
                if len(ppt)==1:
                    pt=''.join(ppt)
                    pt2=''.join(ppt2)
                    od[vb] = od.get(vb,[]) + [[pt,pt2]]
    
    
    with open('05_dir/output_46','a') as f:
        for x in od:
            y=' '.join([ a for a,b in sorted(od[x]) ] + [ b for a,b in sorted(od[x]) ])
            print(x , y , sep='\t' , file=f )            
            
            
for k in range(len(blocks)):
    f07(k)









