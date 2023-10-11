# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 05:39:58 2023

@author: Lenovo
"""

import itertools

def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def generate_rips_complex(points, max_distance):
    rips_complex = {
        "vertices": list(range(len(points))),
        "simplices": [],
        "dimension": -1  # Initialize dimension to -1 and increment it gradually
    }

    for dim in range(len(points)):
        for simplex in itertools.combinations(range(len(points)), dim + 1):
            # Calculate distances between all pairs of points in the simplex
            distances = [euclidean_distance(points[p1], points[p2]) for p1, p2 in itertools.combinations(simplex, 2)]
            if all(distance <= max_distance for distance in distances):
                rips_complex["simplices"].append(list(simplex))
                rips_complex["dimension"] = dim  # Update the highest dimension

    return rips_complex

# Example: Generate a Rips complex and represent it as an abstract simplicial complex
discrete_points = [(0, 0), (2, 1), (2, 1), (3, 3), (3 ,4)]
max_distance = 3.0

rips_complex = generate_rips_complex(discrete_points, max_distance)

print("Abstract Simplicial Complex:")
print("Vertices:", rips_complex["vertices"])
print("Simplices:", rips_complex["simplices"])
print("Highest Dimension:", rips_complex["dimension"])
