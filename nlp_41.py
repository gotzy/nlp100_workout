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
41. 係り受け解析結果の読み込み（文節・係り受け）

40に加えて，文節を表すクラスChunkを実装せよ．

このクラスは
形態素（Morphオブジェクト）のリスト（morphs），
係り先文節インデックス番号（dst），
係り元文節インデックス番号のリスト（srcs）
をメンバ変数に持つこととする．

さらに，入力テキストの係り受け解析結果を読み込み，
１文をChunkオブジェクトのリストとして表現し，
冒頭の説明文の文節の文字列と係り先を表示せよ．

本章の残りの問題では，ここで作ったプログラムを活用せよ．
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

    
for x in blocks[1]:
    print(x[0],x[1].srcs , x[1].dst, [ y.surface for y in x[1].morphs ])





