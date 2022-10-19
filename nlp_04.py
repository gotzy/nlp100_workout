#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html

04. 元素記号
"Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
という文を単語に分解し，
1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，
それ以外の単語は先頭の2文字を取り出し，
取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．
"""

import re


str00="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."


str01=[ re.sub('\.', '', x) for x in str00.split() ]

res={ i+1 : x[:1] if i+1 in (1,5,6,7,8,9,15,16,19)  else x[:2]  for i,x in enumerate(str01) }

print(res)




