# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:04:39 2023

@author: lshen
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:56:58 2023

@author: lshen
"""
import numpy as np
import itertools

def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return np.linalg.norm(np.array(point1)-np.array(point2))

def generate_distance_dic(points):
    distance_dic = {}
    for p1,point1 in enumerate(points):
        for p2,point2 in enumerate(points):
            distance_dic[p1,p2] = euclidean_distance(point1,point2)
    return distance_dic




def generate_rips_complex(pointcloud, max_distance):
    points = [point.tolist() for point in pointcloud]
    rips_complex = {
        "vertices": list(range(len(points))),
        "simplices": [],
        "dimension": -1  # Initialize dimension to -1 and increase it gradually
    }
    distance_dic = generate_distance_dic(points)

    for dim in range(min(len(points),3)):
        for simplex in itertools.combinations(range(len(points)), dim + 1):
            # Calculate distances between all pairs of points in the simplex
            distances = [distance_dic[p1,p2] for p1, p2 in itertools.combinations(simplex, 2)]
            if all(distance <= max_distance for distance in distances):
                rips_complex["simplices"].append(list(simplex))
                rips_complex["dimension"] = dim  # Update the highest dimension

    return rips_complex

from biopandas.mol2 import PandasMol2

mol2file= 'CB7.mol2'
data =PandasMol2().read_mol2(mol2file).df

pointcloud=[]
for x,y,z in zip(data["x"],data["y"],data["z"]):
      pointcloud.append(np.array([x,y,z]))