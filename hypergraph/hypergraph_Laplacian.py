# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 07:45:37 2023

@author: Lenovo
"""
from scipy.linalg import eigh
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


def split_matrix(matrix, index_list):
    # Create an array of boolean values indicating which columns to keep
    keep_columns = np.zeros(matrix.shape[1], dtype=bool)
    keep_columns[index_list] = True

    # Use boolean indexing to split the matrix
    matrix_with_indexed_columns = matrix[:, keep_columns]
    matrix_without_indexed_columns = matrix[:, ~keep_columns]

    return matrix_with_indexed_columns, matrix_without_indexed_columns

def hypergraph_boundary_matrix(hypergraph):
    hypergraph = sorted(hypergraph, key=lambda x: (len(x), x))
    complex=generate(hypergraph)
    num_hyperedges = len(hypergraph) 
    num_simplices = len(complex)
    
    #initiate the index of boundary of hyperedge if in hypergraph
    face_in_hypergraph=set()
    
    #the boundary matrix correspondind to the simplicial closure
    boundary_matrix = np.zeros((num_hyperedges, num_simplices), dtype=int)
    for i, hyperedge in enumerate(hypergraph):
        for j, vertex in enumerate(hyperedge):
            face = hyperedge[:j] + hyperedge[j + 1:]
            if len(face) == 0:
                
                continue

            face_index = complex.index(face)
            if face in hypergraph:
                face_in_hypergraph.add(face_index)
            boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

    return boundary_matrix,face_in_hypergraph

def left_null_space(matrix):
    # Calculate the rank of the input matrix
    rank_matrix = np.linalg.matrix_rank(matrix)

    # Perform SVD to get the left singular vectors
    U, S, VT = np.linalg.svd(matrix)

    # Extract the left null space vectors from U
    left_null_space_vectors = U[:, rank_matrix:].T

    return left_null_space_vectors

def inf_boundary_matrix(hypergraph):
    hypergraph = sorted(hypergraph, key=lambda x: (len(x), x))
    complex=generate(hypergraph)
    boundary_matrix,index_set=hypergraph_boundary_matrix(hypergraph)  
    num_hyperedges = len(hypergraph) 
    m = len(hypergraph[num_hyperedges - 1]) - 1
    
    #compute the index of dimensions of simplices
    b = dim_index(complex)

    # #indices for hyperegdes
    # indices = [i for i, x in enumerate(complex) if x in hypergraph]
    # print(indices)    
    #compute the index of dimensions of hyperedes
    c = dim_index(hypergraph)

    #initiate the boundary matrix in dimension 0
    boundary_0 = np.zeros((c[1] - c[0], 0), dtype=int)
    
    #compute the boundary matrices in different dimensions
    boundary = [np.zeros((0, b[i + 1] - b[i]), dtype=int) if c[i + 2] - c[i + 1] == 0
     else boundary_matrix[c[i + 1]:c[i + 2], b[i]:b[i + 1]]
     for i in range(m)]
    
    #compute the boundary matrix of infimum complex
    inf_boundary=[]
    #rank_inf count the rank of infimum complex in each dimension
    rank_inf=c[1]-c[0]
    #A_inv is iterative, and we use the A_inv in the last iteration
    A_inv=np.eye(rank_inf)
    for i in range(m):

        if boundary[i].size==0:
            inf_boundary=inf_boundary+[np.zeros((0, rank_inf),dtype=int)]
            rank_inf=0
        else:
            index_set_i = set()
            for x in index_set:
                if x >= b[i]-b[0] and x<b[i+1]-b[0]:
                    index_set_i.add(x-b[i])
            index_i=list(index_set_i)
            #two new matrices by removing the columns indexed by the list and not indexed by the list.
            #B[1] is \bar{B}, B[0] is \tilde{B}
            B = split_matrix(boundary[i],index_i)

            if B[1].size==0 or np.all(B[1] == 0):
                inf_boundary=inf_boundary+[boundary[i]@A_inv]
                rank_inf=boundary[i].shape[0]
                A_inv=np.eye(rank_inf)
            else:
                A = left_null_space(B[1])

                if rank_inf==0:
                    inf_boundary=inf_boundary+[np.zeros((A.shape[0], rank_inf),dtype=int)]
                else:
                    inf_boundary=inf_boundary+[A@B[0]@A_inv] 
                    
                rank_inf=A.shape[0]
                A_inv=np.linalg.pinv(A)

    inf_boundary=[boundary_0]+inf_boundary     
    return inf_boundary

def hypergraph_laplacian_matrix(hypergraph):
    # Compute the boundary matrix of infimum complex
    inf_boundary = inf_boundary_matrix(hypergraph)  
    # Compute the Laplacian matrix by multiplying the Dirac matrix by its transpose
    length=len(inf_boundary)

    initial_matrix=np.dot(inf_boundary[0], inf_boundary[0].T)
    if length>1:
        initial_matrix = initial_matrix+np.dot(inf_boundary[1].T, inf_boundary[1])   
    
    laplacian_matrix = [initial_matrix]
    for i in range(1,length-1):

        laplacian_matrix = laplacian_matrix + [np.dot(inf_boundary[i], inf_boundary[i].T)+np.dot(inf_boundary[i+1].T, inf_boundary[i+1])]
    if length>1:
        laplacian_matrix = laplacian_matrix +[np.dot(inf_boundary[length-1], inf_boundary[length-1].T)]

    return laplacian_matrix


#hypergraph = [[2], [3], [2, 3, 5], [2, 4, 5], [3, 4, 5], [2, 3, 4]]
#hypergraph = [[3, 4, 5], [2, 3, 4],[2,3,4,5,6,7]]
hypergraph = [[0],[1],[2],[3],[4],[5],
              [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
              [0, 1 , 2],[1, 2 , 3],[2, 3, 4],[3, 4, 5],[0, 4, 5], [0, 1 , 5]]
# hypergraph = [[0],[1],[2],[3],[4],[5],
#               [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
#               [0, 1 , 2],[1, 2 , 3],[2, 3, 4],[3, 4, 5],[0, 4, 5], [0, 1 , 5],
#               [0, 1 , 3],[0, 1 , 4],[0, 2, 3],[0, 2, 5],[0, 3, 4], [0, 3 , 5], 
#               [1, 2 , 4],[1, 2 , 5],[1, 3, 4],[1, 4, 5],[2, 3, 5], [2, 4 , 5], 
#                 ]
# hypergraph = [[1, 2], [1, 2, 3], [1, 3], [1, 2, 3, 4], [1, 2, 4], 
#               [1, 3, 4], [2, 3], [2, 3, 4], [2, 4], [3, 4]]
#hypergraph = [[1, 2], [2, 1],[1,2,1],[2,1,2]]
# hypergraph =[[1, 3], [2, 4], [1, 2], [2], [3, 4], [1, 2, 3], [4], 
#              [1, 3, 4], [2, 3], [1], [2, 3, 4], [3], [1, 2, 4]]
# Calculate the Laplacian matrix using the function
#boundary_matrix = hypergraph_boundary_matrix(hypergraph)
#inf_boundary=inf_boundary_matrix(hypergraph)
#print("boundary_matrix:",inf_boundary)
hypergraph_laplacian = hypergraph_laplacian_matrix(hypergraph)
# Print the Laplacian matrix
print("Laplacian Matrix:",hypergraph_laplacian)

# 计算Laplacian矩阵的特征值和特征向量
eigenvalues, eigenvectors = eigh(hypergraph_laplacian[1])
# 特征值存储在eigenvalues数组中
print("特征值：", eigenvalues)