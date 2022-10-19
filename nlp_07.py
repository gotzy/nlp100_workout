#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html

07. テンプレートによる文生成
引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．
さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．
"""



def f01(x_, y_, z_):
    return '{0}時の{1}は{2}'.format(x_ , y_ , z_)

x=12
y="気温"
z=22.4


print(f01(x,y,z))



