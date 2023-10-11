import networkx as nx
import Laplacian_function


# Create a directed graph (digraph)
G = nx.DiGraph()

# Add edges to the digraph
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)
G.add_edge(2, 4)
G.add_edge(3, 4)

# Initialize paths with vertices as paths of length 0
paths = [[node] for node in G.nodes()]

# Function to find all paths of a given length in a digraph
def find_paths_of_length(graph, length):
    found_paths = []
    for start_node in graph.nodes():
        for end_node in graph.nodes():
            if start_node != end_node:
                found_paths.extend(list(nx.all_simple_paths(graph, start_node, end_node, cutoff=length)))
    return found_paths

# Find all paths of lengths 1 and 2
for path_length in range(1, 3):
    found_paths = find_paths_of_length(G, path_length)
    paths.extend(found_paths)

# Use a set to eliminate duplicates and then convert it back to a list
unique_paths = list(map(list, set(map(tuple, paths))))

print(unique_paths)



path_laplacian =Laplacian_function.hypergraph_laplacian_matrix(unique_paths)
print("Laplacian Matrix:",path_laplacian)