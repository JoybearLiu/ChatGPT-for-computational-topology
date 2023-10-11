# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:34:36 2023

@author: Lenovo
"""

# import numpy as np

# def compute_boundary_matrix(complex):
#     num_simplices = len(complex)

#     # Initialize the boundary matrix with zeros
#     boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

#     for i, simplex in enumerate(complex):
#         for j, vertex in enumerate(simplex):
#             face = simplex[:j] + simplex[j + 1:]
#             if len(face) == 0:
#                 continue

#             face_index = complex.index(face)
#             boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

#     return boundary_matrix

# # Define the abstract simplicial complex as a list of simplices
# complex = [[1], [2], [3], [4], [5], [1, 2], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

# # Calculate the boundary matrix using the function
# boundary_matrix = compute_boundary_matrix(complex)

# # Print the boundary matrix
# print("Boundary Matrix:")
# print(boundary_matrix)


# import numpy as np

# def compute_boundary_matrix(complex):
#     num_simplices = len(complex)

#     # Initialize the boundary matrix with zeros
#     boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

#     for i, simplex in enumerate(complex):
#         for j, vertex in enumerate(simplex):
#             face = simplex[:j] + simplex[j + 1:]
#             if len(face) == 0:
#                 continue

#             face_index = complex.index(face)
#             boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

#     return boundary_matrix

# def compute_dirac_matrix(complex):
#     # Compute the boundary matrix
#     boundary_matrix = compute_boundary_matrix(complex)
    
#     # Compute the Dirac matrix by adding the boundary matrix and its transpose
#     dirac_matrix = boundary_matrix + boundary_matrix.T

#     return dirac_matrix

# # Define the abstract simplicial complex as a list of simplices
# complex = [[1], [2], [3], [4], [5], [1, 2], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

# # Calculate the Dirac matrix using the function
# dirac_matrix = compute_dirac_matrix(complex)

# # Print the Dirac matrix
# print("Dirac Matrix:")
# print(dirac_matrix)


import numpy as np

def compute_boundary_matrix(complex):
    num_simplices = len(complex)

    # Initialize the boundary matrix with zeros
    boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

    for i, simplex in enumerate(complex):
        for j, vertex in enumerate(simplex):
            face = simplex[:j] + simplex[j + 1:]
            if len(face) == 0:
                continue

            face_index = complex.index(face)
            boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

    return boundary_matrix

def compute_dirac_matrix(complex):
    # Compute the boundary matrix
    boundary_matrix = compute_boundary_matrix(complex)
    
    # Compute the Dirac matrix by adding the boundary matrix and its transpose
    dirac_matrix = boundary_matrix + boundary_matrix.T

    return dirac_matrix

def compute_laplacian_matrix(complex):
    # Compute the Dirac matrix
    dirac_matrix = compute_dirac_matrix(complex)
    
    # Compute the Laplacian matrix by multiplying the Dirac matrix by its transpose
    laplacian_matrix = np.dot(dirac_matrix, dirac_matrix.T)

    return laplacian_matrix

# Define the abstract simplicial complex as a list of simplices
complex = [[0], [1], [2], [3], [4], [5], [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [1, 5], [2, 3], [2, 5], [3, 4], [3, 5], [4, 5], [0, 1, 2], [0, 1, 4], [0, 2, 3], [0, 3, 4], [1, 2, 5], [1, 4, 5], [2, 3, 5], [3, 4, 5]]

# boundary_matrix = compute_boundary_matrix(complex)
# print(boundary_matrix)


# Calculate the Laplacian matrix using the function
laplacian_matrix = compute_laplacian_matrix(complex)

# Print the Laplacian matrix
print("Laplacian Matrix:")
print(laplacian_matrix)


