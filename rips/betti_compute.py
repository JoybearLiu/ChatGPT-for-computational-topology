# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:58:55 2023

@author: Lenovo
"""

# import numpy as np

# # Define the abstract simplicial complex as a list of simplices
# complex = [[1], [2], [3], [4], [5], [1, 2], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

# # Extract the unique vertices from the simplicial complex
# vertices = list(set([v for simplex in complex for v in simplex]))
# vertices.sort()  # Sort vertices for consistent ordering

# # Create a dictionary to map vertices to their indices
# vertex_to_index = {v: i for i, v in enumerate(vertices)}

# # Determine the number of vertices and simplices
# num_vertices = len(vertices)
# num_simplices = len(complex)

# # Initialize the boundary matrix with zeros
# boundary_matrix = np.zeros((num_simplices, num_vertices), dtype=int)

# # Fill in the boundary matrix
# for i, simplex in enumerate(complex):
#     for j, vertex in enumerate(simplex):
#         # Remove the j-th vertex from the simplex to get the (j-1)-face
#         face = simplex[:j] + simplex[j + 1:]
#         # Check if the face is empty (0-simplex)
#         if len(face) == 0:
#             continue
        
#         # Find the index of the face in the vertex list
#         face_index = vertex_to_index[face[0]]
        
#         # Add 1 to the (i, face_index) entry of the boundary matrix
#         boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

# # Print the boundary matrix
# print("Boundary Matrix:")
# print(boundary_matrix)


# import numpy as np

# # Define the abstract simplicial complex as a list of simplices
# complex = [[1], [2], [3], [4], [5], [1, 2], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

# # Determine the number of vertices and simplices
# num_simplices = len(complex)
# m=len(complex[num_simplices-1])-1

# b = [0, 0]  
# k = 1  

# for i in range(num_simplices):
#     if len(complex[i]) == k :
#         b[k] += 1
#     else:
#         k += 1
#         b.append(b[k-1]+1)
# # Initialize the boundary matrix with zeros
# boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

# # Fill in the boundary matrix
# for i, simplex in enumerate(complex):
#     for j, vertex in enumerate(simplex):
#         # Remove the j-th vertex from the simplex to get the (j-1)-face
#         face = simplex[:j] + simplex[j + 1:]
#         # Check if the face is empty (0-simplex)
#         if len(face) == 0:
#             continue
        
#         # Find the index of the face in the simplex list
#         face_index = complex.index(face)
#         # Add 1 to the (i, face_index) entry of the boundary matrix
#         boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

# # Print the boundary matrix
# print("Boundary Matrix:")
# print(boundary_matrix)
# r=[0]*(m+1)
# for i in range(m):
#     r[i+1]=np.linalg.matrix_rank(boundary_matrix[b[i+1]:b[i+2],b[i]:b[i+1]])

# betti=[0]*(m+1)
# for i in range(m):
#     betti[i]=b[i+1]-b[i]-r[i]-r[i+1]
# print(betti)





# import numpy as np

# def compute_betti(complex):
#     num_simplices = len(complex)
#     m = len(complex[num_simplices - 1]) - 1

#     b = [0, 0]
#     k = 1

#     for i in range(num_simplices):
#         if len(complex[i]) == k:
#             b[k] += 1
#         else:
#             k += 1
#             b.append(b[k - 1] + 1)

#     boundary_matrix = np.zeros((num_simplices, num_simplices), dtype=int)

#     for i, simplex in enumerate(complex):
#         for j, vertex in enumerate(simplex):
#             face = simplex[:j] + simplex[j + 1:]
#             if len(face) == 0:
#                 continue

#             face_index = complex.index(face)
#             boundary_matrix[i, face_index] = 1 if j % 2 == 0 else -1

#     r = [0] * (m + 1)
#     for i in range(m):
#         r[i + 1] = np.linalg.matrix_rank(boundary_matrix[b[i + 1]:b[i + 2], b[i]:b[i + 1]])

#     betti = [0] * (m + 1)
#     for i in range(m):
#         betti[i] = b[i + 1] - b[i] - r[i] - r[i + 1]

#     return betti

# # Define the abstract simplicial complex as a list of simplices
# complex = [[1], [2], [3], [4], [5]]#, [1, 2], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

# # Calculate Betti numbers using the function
# betti_numbers = compute_betti(complex)

# # Print the Betti numbers
# print("Betti Numbers:")
# print(betti_numbers)


#debug version
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
    print(r)
    print(b)    
    print(m)
    betti = [0] * (m + 1)
    if m==0:
        betti[0]=b[1] - b[0]
    elif m==1:      
        betti[0]=b[1] - b[0] - r[0] - r[1]
        betti[m]=b[2] - b[1] - r[1]
    else:
        for i in range(m):
            betti[i] = b[i + 1] - b[i] - r[i] - r[i + 1]

    return betti

# Define the abstract simplicial complex as a list of simplices
#complex = [[1], [2], [3], [4], [5], [1, 2], [2, 3], [2, 4], [3, 4]]#, [2, 3, 4]]
complex =[[0], [1], [2], [3], [4], [5], [0, 1], [0, 2], [1, 4], [2, 3], [3, 5], [4, 5]]
complex = [[0],[1],[2],[3],[4],[5],
              [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]]
# Calculate Betti numbers using the function
betti_numbers = compute_betti(complex)

# Print the Betti numbers
print("Betti Numbers:")
print(betti_numbers)
