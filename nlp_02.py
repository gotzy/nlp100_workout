#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html

02. 「パトカー」＋「タクシー」＝「パタトクカシーー」
「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．
"""

str00="パトカー"
str01="タクシー"

str02 = "".join([  x+y  for x,y in zip(str00,str01) ])

print(str02)


