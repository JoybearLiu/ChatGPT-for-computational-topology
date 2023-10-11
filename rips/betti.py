# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:09:13 2023

@author: lshen
"""

import numpy as np

def compute_betti(complex):
    num_simplices = len(complex)
    m = len(complex[num_simplices - 1]) - 1

    b = [0, 0]
    k = 1

    for i in range(num_simplices):
        if len(complex[i]) == k:
            b[k] += 1
        else:
            k += 1
            b.append(b[k - 1] + 1)

    boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

    for i, simplex in enumerate(complex):
        for j, vertex in enumerate(simplex):
            face = simplex[:j] + simplex[j + 1:]
            if len(face) == 0:
                continue

            face_index = complex.index(face)
            boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

    r = [0] * (m + 1)
    for i in range(m):
        r[i + 1] = np.linalg.matrix_rank(boundary_matrix[b[i + 1]:b[i + 2], b[i]:b[i + 1]])

    betti = [0] * (m + 1)
    if m==0:
        betti[0]=b[1] - b[0]
    elif m==1:
        betti[0]=b[1] - b[0]
        betti[m]=b[2] - b[1] - r[1]
    else:
        for i in range(m):
            betti[i] = b[i + 1] - b[i] - r[i] - r[i + 1]

    return betti