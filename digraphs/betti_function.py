# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:46:24 2023

@author: Lenovo
"""

import numpy as np
import itertools


def powerset(iterable):
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1)))


def generate(cell_array):
    # Set to store the results
    result_set = set()

    # Iterate through each list in cell_array
    for lst in cell_array:
        # Generate non-empty subsets and add them to the result set
        non_empty_subsets = powerset(lst)
        result_set.update(non_empty_subsets)

    # Convert the result set into a list
    merged_result = [list(subset) for subset in result_set]

    # Sort merged_result using a composite key: first by length, then by lexicographic order
    merged_result.sort(key=lambda x: (len(x), x))

    return merged_result



def find_indices_of_subset(main_set, subset):
    indices = [i for i, x in enumerate(main_set) if x in subset]
    return indices



def dim_index(hypergraph):
    num_hyperedges = len(hypergraph) 
    m=len(hypergraph[num_hyperedges-1])+1
    b = [0] * m
    for i in range(num_hyperedges):
        b[len(hypergraph[i])]+=1;  
    c = [0] * m
    for i in range(m-1):
        c[i+1]=c[i]+b[i+1]
    
    return c


def compute_betti_hypergraph(hypergraph):
    #compute the simplicial closure of the hypergraph
    hypergraph = sorted(hypergraph, key=lambda x: (len(x), x))
    complex=generate(hypergraph)
    num_hyperedges = len(hypergraph) 
    num_simplices = len(complex)
    m = len(hypergraph[num_hyperedges - 1]) - 1
    
    #compute the index of dimensions of simplices
    b = dim_index(complex)
    
    #compute the index of dimensions of hyperedes
    c = dim_index(hypergraph)
    
    #the boundary matrix correspondind to the simplicial closure
    boundary_matrix = np.zeros((num_hyperedges, num_simplices), dtype=int)
    for i, hyperedge in enumerate(hypergraph):
        for j, vertex in enumerate(hyperedge):
            face = hyperedge[:j] + hyperedge[j + 1:]
            if len(face) == 0:
                continue

            face_index = complex.index(face)
            boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1
    
    #index the hyperedges of the hypergraph in the simplicial closure
#    indices= find_subset_indices(complex, hypergraph)
    indices= find_indices_of_subset(complex, hypergraph)
    
    hyperedge_matrix=np.zeros((num_hyperedges,num_simplices), dtype=int)
    for j in range(num_hyperedges):
        hyperedge_matrix[j,indices[j]]=1
       
    #compute the Betti numbers
    r = [0] * (m + 1) 
    union=[0] * (m + 1) 
    for i in range(m):
        # it would be an error if we use rank directly here
        if boundary_matrix[c[i + 1]:c[i + 2],b[i]:b[i + 1]].size==0:
            r[i + 1] = 0
        else:
            r[i + 1] = np.linalg.matrix_rank(boundary_matrix[c[i + 1]:c[i + 2],b[i]:b[i + 1]])
      
        #compute the rank of combined matrix
        combined_matrix = np.vstack((hyperedge_matrix[c[i]:c[i + 1],b[i]:b[i + 1]],boundary_matrix[c[i + 1]:c[i + 2],b[i]:b[i + 1]]))
        # it would be an error if we use rank directly here
        if combined_matrix.size==0:
            union[i]=0
        else:
            union[i]=np.linalg.matrix_rank(combined_matrix)             
    betti = [0] * (m + 1)

    for i in range(m):
        betti[i] = union[i] - r[i] - r[i + 1]
    if m==0:
        betti[m]=c[1]-c[0];
    else:
        betti[m]=np.linalg.matrix_rank(hyperedge_matrix[c[m]:c[m + 1],b[m]:b[m + 1]])-np.linalg.matrix_rank(boundary_matrix[c[m]:c[m + 1],b[m-1]:b[m]])
    return betti


