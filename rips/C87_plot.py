# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from cb7_rips import generate_rips_complex
from betti import compute_betti
from fast_compute import compute_laplacian_matrix

def get_betti_lap(pointcloud,distances):
    betti0 = []
    betti1 = []
    lap0 = []
    lap1 = []
    e0=[]
    e1=[]
    import time
    for distance in distances:
        print(distance)
        st = time.time()
        rips_complex = generate_rips_complex(pointcloud, distance)
        betti = compute_betti(rips_complex["simplices"])
        et = time.time()
        print(len(rips_complex["simplices"]),et-st)
        betti0.append(betti[0])
        betti1.append(0 if len(betti)==1 else betti[1])
        laplacian = compute_laplacian_matrix(rips_complex["simplices"])
        if laplacian[0] == []:
            lap0.append(0)
        else:
            eigval0 = np.linalg.eigvalsh(laplacian[0])
            p_e0 = eigval0[eigval0>1e-10]
            e0.append(p_e0)
            lap0.append(np.min(p_e0) if p_e0.size > 0 else 0)
        if type(laplacian[1]) == type(None):
            lap1.append(0)
        else:
            eigval1 = np.linalg.eigvalsh(laplacian[1])
            p_e1 = eigval1[eigval1>1e-10]
            e1.append(p_e1)
            lap1.append(np.min(p_e1) if p_e1.size > 0 else 0)
        et = time.time()
        print("done",et-st)
    
    return betti0,betti1,lap0,lap1,e0,e1



def plot_persistent_graph(distances,value,axe):
    points_x =[]
    points_y=[]
    for i,x in enumerate(distances):
        points_x.append(x)
        points_y.append(value[i])
        if i <len(distances)-1:
            points_x.append(distances[i+1])
            points_y.append(value[i])
            
    axe.plot(points_x,points_y)




#from biopandas.mol2 import PandasMol2

# mol2file= 'CB7.mol2'
# data =PandasMol2().read_mol2(mol2file).df

# pointcloud=[]
# for x,y,z in zip(data["x"],data["y"],data["z"]):
#      pointcloud.append(np.array([x,y,z]))
#distances=[0.4,1.13,1.42,1.59,1.6,2.42,2.63,3.6]


def read_xyz(path):
    pointcloud = []
    with open(path,'r') as f:
        for item in f.readlines():
            print(item.split('  '))
            x = float(item.split()[0])
            y = float(item.split()[1])
            z = float(item.split()[2])
            pointcloud.append(np.array([x,y,z],dtype=np.float32))
        f.close()
            
    return pointcloud


pointcloud = read_xyz('./C20.xyz')
from scipy.spatial import distance_matrix

radius = np.linspace(0,4,401)
distances = 2*radius






b0,b1,l0,l1,e0,e1 = get_betti_lap(pointcloud,distances)


    










fig = plt.figure()
axes = fig.subplots(nrows=2, ncols=2)
plot_persistent_graph(radius, b0, axes[0,0])
plot_persistent_graph(radius, b1, axes[0,1])
plot_persistent_graph(radius, l0, axes[1,0])
plot_persistent_graph(radius, l1, axes[1,1])

