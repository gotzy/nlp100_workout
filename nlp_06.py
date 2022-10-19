#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html

06. 集合
“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，
それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．
"""

str01="paraparaparadise"
str02="paragraph"


def f01( wd , n ):
    return { wd[i:i+n]  for i in range(len(wd) - n + 1 ) }


# 文字bi-gram
X=f01(str01,2)
Y=f01(str02,2)

# 和集合
print( 'X | Y :  ' , X | Y )

# 積集合
print( 'X & Y :  ' , X & Y )

# 差集合
print( 'X - Y :  ' , X - Y )
print( 'Y - X :  ' , Y - X )

# 'se' check
print( "'se' in X  : " , 'se' in X )
print( "'se' in Y  : " , 'se' in Y )






