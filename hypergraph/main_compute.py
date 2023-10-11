# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:51:31 2023

@author: lshen
"""
import numpy as np
from hypergraph_betti import compute_betti_hypergraph
from hypergraph_Laplacian import hypergraph_laplacian_matrix
hypergraph1 = [
    [0],[1],[2],[3],[4],[5]
    ]
hypergraph2 = [
               [0],[1],[2],[3],[4],[5],
               [0, 1],[1, 2], [2, 3], [3, 4], [4, 5], [0, 5]         
               ]
hypergraph3 = [
               [0],[1],[2],[3],[4],[5],
               [0, 1],[1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
               [0, 1 , 2],[1, 2 , 3],[2, 3, 4],[3, 4, 5],[0, 4, 5], [0, 1 , 5]
               ]

hypergraph4 = [
               [0],[1],[2],[3],[4],[5],
               [0, 1],[1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
               [0, 1, 2],[1, 2, 3],[2, 3, 4],[0,4,5],[0,1,5],[0,1,3],[0,1,4],[0,2,3],[0,2,5],
                [0,3,4],[0,3,5],[1,2,4],[1,2,5],[1,3,4],[1,4,5],[2,3,5],[2,4,5],[3, 4, 5]
               ]

hypergraphs = [hypergraph1,hypergraph2,hypergraph3,hypergraph4]

def get_minimal_numda(laplacian_matrix):
    eigvals = np.linalg.eigvalsh(laplacian_matrix)
    p_e = eigvals[eigvals>0]
    if p_e.size == 0:
        return 0
    else:
        return p_e[0]
B = []
L=[]
for hypergraph in hypergraphs:
    B.append(compute_betti_hypergraph(hypergraph))
    laps = hypergraph_laplacian_matrix(hypergraph)
    numda = [get_minimal_numda(lap) for lap in laps]
    L.append(numda)
    

    