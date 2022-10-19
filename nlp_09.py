#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html


09. Typoglycemia

スペースで区切られた単語列に対して，
各単語の先頭と末尾の文字は残し，
それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
ただし，長さが４以下の単語は並び替えないこととする．

適当な英語の文
（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，
その実行結果を確認せよ．
"""


import random


def typoglycemia(sentence):
    
    ls_res=list()
    
    for wd in sentence.split() :
        if len(wd)>=5:
            random.seed(123)
            a = list(wd[:1])
            b = list(wd[1:-1])
            c = list(wd[-1:])
            random.shuffle(b)
            res_wd = ''.join( a+b+c )
        else:
            res_wd = wd
        
        ls_res.append(res_wd)
    
    return ' '.join(ls_res)



sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."


print(typoglycemia( sentence ))





