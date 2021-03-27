#!/usr/bin/env python3
import string
from collections import defaultdict
from fractions import Fraction as F
from queue import SimpleQueue
from random import randint

W = 30
H = 30

SHIFT = 3
L_COEFF = []

ST = [
    (
        2,
        [
            [0, 1, 2, 1, 0],
            [1, 3, 4, 3, 1],
            [2, 4, 5, 4, 2],
            [1, 3, 4, 3, 1],
            [0, 1, 2, 1, 0],
        ],
        [
            F(0),
            F(-1, 30),
            F(-1, 60),
            F(4, 15),
            F(13, 15),
            F(-21, 5),
        ],
    ),
    (
        3,
        [
            [0, 0, 1, 2, 1, 0, 0],
            [0, 3, 4, 5, 4, 3, 0],
            [1, 4, 6, 7, 6, 4, 1],
            [2, 5, 7, 8, 7, 5, 2],
            [1, 4, 6, 7, 6, 4, 1],
            [0, 3, 4, 5, 4, 3, 0],
            [0, 0, 1, 2, 1, 0, 0],
        ],
        [
            F(0),
            F(-17, 180),
            F(1, 45),
            F(-29, 180),
            F(47, 45),
            F(7, 30),
            F(-187, 90),
            F(-191, 45),
            F(779, 45),
        ],
    ),
]

dep = defaultdict(list)

for i in range(W):
    for j in range(H):
        res = []
        for s, st, coeffs in ST:
            for di in range(-s, s + 1):
                for dj in range(-s, s + 1):
                    val = st[di + s][dj + s]
                    if val == 0:
                        continue
                    ni = (i + di + W) % W
                    nj = (j + dj + H) % H
                    val = coeffs[val]
                    res.append((ni, nj, val))

        dep[(i, j)] += res


matrix = [[0 for _ in range(W * H)] for _ in range(W * H)]
for coord1 in dep:
    a = coord1[0] * W + coord1[1]
    for coord2 in dep[coord1]:
        b = coord2[0] * W + coord2[1]
        matrix[a][b] += coord2[2]

for line in matrix:
    total = 0
    for val in line:
        if val != 0:
            total += 1
    print(total)


s = string.ascii_letters + string.digits
c_i = 0
chars = {}
for line in matrix:
    for val in line:
        if val == 0:
            c = "."
        elif val in chars:
            c = chars[val]
        else:
            c = s[c_i]
            c_i = (c_i + 1) % len(s)
            chars[val] = c
        print(c, end="")
    print()
