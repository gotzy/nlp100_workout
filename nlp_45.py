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
45. 動詞の格パターンの抽出Permalink
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． 
ただし，出力は以下の仕様を満たすようにせよ．

動詞を含む文節において，最左の動詞の基本形を述語とする
述語に係る助詞を格とする
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
この文は「作り出す」という１つの動詞を含み，
「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると解析された場合は，
次のような出力になるはずである．
--
作り出す	で は を
--
このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．
 * コーパス中で頻出する述語と格パターンの組み合わせ
 * 「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）

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

with open('05_dir/output_45','w') as f:
    pass
    
def f06(nn_):
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
                if len(ppt)==1:
                    pt=''.join(ppt)
                    od[vb] = od.get(vb,[]) + [pt]

    with open('05_dir/output_45','a') as f:
        for x in od:
            print(x , ' '.join(sorted(list(set(od[x])))) , sep='\t' , file=f )            
            
            
for k in range(len(blocks)):
    f06(k)



### unix command

## 1
# cat 05_dir/output_45 | sort | uniq -c | sort -k1 -nr | head -n10

## 2a
# cat 05_dir/output_45 | grep '行う' | sort | uniq -c | sort -nr | head -n10

## 2b
# cat 05_dir/output_45 | grep 'なる' | sort | uniq -c | sort -nr | head -n10

## 2c
# cat 05_dir/output_45 | grep '与える' | sort | uniq -c | sort -nr | head -n10














