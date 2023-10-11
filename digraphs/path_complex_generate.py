# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:22:38 2023

@author: Lenovo
"""

import networkx as nx

# Create a directed graph (digraph)
G = nx.DiGraph()

# Add edges to the digraph
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)
G.add_edge(2, 4)
G.add_edge(3, 4)

# Function to find all paths in a digraph
def find_all_paths(graph):
    all_paths = []
    for start_node in graph.nodes():
        for end_node in graph.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(graph, start_node, end_node))
                all_paths.extend(paths)
    return all_paths

all_paths = find_all_paths(G)
print("All directed paths in the digraph are:")
for path in all_paths:
    print(path)
