#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nlp100.github.io/ja/ch01.html


08. 暗号文
与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．

英小文字ならば(219 - 文字コード)の文字に置換
その他の文字はそのまま出力
この関数を用い，英語のメッセージを暗号化・復号化せよ．
"""



def cipher(wd):
    return ''.join([ chr(219 - ord(x)) if x.islower() else x  for x in wd ])


print(cipher( 'This is a pen.' ))

print(cipher( 'I read a book.' ))





