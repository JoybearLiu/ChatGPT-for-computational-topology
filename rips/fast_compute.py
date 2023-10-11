# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:48:31 2023

@author: Lenovo
"""

import numpy as np

import time
start_time = time.time()


#Used for computing the index for the 0,1,2-th boundary matrix
def dim_index(complex):
    num_simplices = len(complex)
    m = 2
    k = 1
    b = [0,0,0,0]

    for i in range(num_simplices):
        if len(complex[i]) == k:
            b[k] += 1
        else:
            k += 1
            if k > m+1:
                break   
            b[k]=b[k-1]+1
    
    return b


def compute_boundary_matrix(complex):
    num_simplices = len(complex)
    
    #Used for computing the index for the 0,1,2-th boundary matrix
    b=dim_index(complex)
    print(b)
    
    
    # Initialize the boundary matrix with zeros
    boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

    for i, simplex in enumerate(complex):
        if i>b[3] and (b[3]>0):
            break
        for j, vertex in enumerate(simplex):
            face = simplex[:j] + simplex[j + 1:] 
            if len(face) == 0:
                continue
            face_index = complex.index(face)
            boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1
    print(boundary_matrix)      
    #    boundary_matrix = compute_boundary_matrix(complex)
    Lap_0=np.zeros((b[1]-b[0], b[1]-b[0]), dtype=int)  
    boundary_1=np.array([])
    boundary_2=np.array([])
    if b[2]>0:
        boundary_1 = boundary_matrix[b[1]:b[2], b[0]:b[1]]         
    if b[3]>0:
        boundary_2 = boundary_matrix[b[2]:b[3], b[1]:b[2]]

    # Create a list containing boundary_1 and boundary_2
    boundary_matrices = [Lap_0,boundary_1, boundary_2]   
    return boundary_matrices



# Compute the Laplacian matrix
def compute_laplacian_matrix(complex):
    boundary =compute_boundary_matrix(complex)
    print(boundary)  
    laplacian_matrix_1=boundary[0]+np.dot(boundary[1].T, boundary[1])
    if boundary[1].size==0:
        laplacian_matrix_2=None 
    else:
        laplacian_matrix_2=np.dot(boundary[1], boundary[1].T)+np.dot(boundary[2].T, boundary[2])
    laplacian_matrices=[laplacian_matrix_1,laplacian_matrix_2]
    return laplacian_matrices
    

# Define the abstract simplicial complex as a list of simplices

#complex = [[1], [2], [3], [4], [5], [1, 2], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
#complex =[[0], [1], [2], [3], [4], [5], [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [1, 5], [2, 3], [2,4],[2, 5], [3, 4], [3, 5], [4, 5], [0, 1, 2], [0, 1, 4], [0, 2, 3], [0, 3, 4], [1, 2, 5], [1, 4, 5], [2,3,4],[2, 3, 5],[2,4,5], [3, 4, 5],[2,3,4,5]]
#complex =[[0], [1], [2], [3], [4], [5], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5], [0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 4, 5], [0, 1, 3, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
laplacian_operator=compute_laplacian_matrix(complex)
#e=np.linalg.eigvalsh(laplacian_operator[1])

# Print the Laplacian matrix
print("Laplacian Matrices:")

print(laplacian_operator)
#print(e)
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")