#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html

05. n-gram
与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．
"""

str01="I am an NLPer"


def f01( wd , n ):
    return [ wd[i:i+n]  for i in range(len(wd) - n + 1 ) ]

# 単語bi-gram
print( f01( str01.split() , 2 ) )


# 文字bi-gram
print(f01(str01,2))




