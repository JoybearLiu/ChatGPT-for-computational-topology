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
    # 计算两点之间的欧几里得距离
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
        "dimension": -1  # 初始维度设为-1，之后将其逐渐增加
    }
    distance_dic = generate_distance_dic(points)

    for dim in range(min(len(points),3)):
        for simplex in itertools.combinations(range(len(points)), dim + 1):
            # 计算simplex中所有点对之间的距离
            distances = [distance_dic[p1,p2] for p1, p2 in itertools.combinations(simplex, 2)]
            if all(distance <= max_distance for distance in distances):
                rips_complex["simplices"].append(list(simplex))
                rips_complex["dimension"] = dim  # 更新最高维度

    return rips_complex
