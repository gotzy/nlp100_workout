#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html

03. 円周率
“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”
という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．
"""

import re


str00="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

res=[ len(re.sub('\.|\,', '', x)) for x in str00.split() ]

print(res)


