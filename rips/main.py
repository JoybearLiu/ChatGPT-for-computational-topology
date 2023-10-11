# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from rips import generate_rips_complex
from betti import compute_betti
from laplacian import compute_laplacian_matrix

def get_betti_lap(pointcloud,distances):
    betti0 = []
    betti1 = []
    lap0 = []
    lap1 = []

    import time
    previous_complex_number = 0
    for distance in distances:
        print(distance)
        st = time.time()
        rips_complex = generate_rips_complex(pointcloud, distance)
        if len(rips_complex["simplices"]) == previous_complex_number:
            betti0.append(betti0[-1])
            betti1.append(betti1[-1])
            lap0.append(lap0[-1])
            lap1.append(lap1[-1])
        else:
            
            betti = compute_betti(rips_complex["simplices"])
        
            betti0.append(betti[0])
            betti1.append(0 if len(betti)==1 else betti[1])
            laplacian = compute_laplacian_matrix(rips_complex["simplices"])
            if laplacian[0] == []:
                lap0.append(0)
            else:
                eigval0 = np.linalg.eigvalsh(laplacian[0])
                p_e0 = eigval0[eigval0>1e-10]
                lap0.append(np.min(p_e0) if p_e0.size > 0 else 0)
            if type(laplacian[1]) == type(None):
                lap1.append(0)
            else:
                eigval1 = np.linalg.eigvalsh(laplacian[1])
                p_e1 = eigval1[eigval1>1e-10]
                lap1.append(np.min(p_e1) if p_e1.size > 0 else 0)
        previous_complex_number = len(rips_complex["simplices"])
        et = time.time()
        print("done",et-st)
    
    return betti0,betti1,lap0,lap1




## computation plot for CB7
from biopandas.mol2 import PandasMol2

mol2file= 'CB7.mol2'
data =PandasMol2().read_mol2(mol2file).df

pointcloud=[]
for x,y,z in zip(data["x"],data["y"],data["z"]):
      pointcloud.append(np.array([x,y,z]))

distances=[0.4,1.13,1.42,1.59,1.6,2.42,2.63,3.6]
radius = np.linspace(0,2,201)
distances = 2*radius

b0,b1,l0,l1 = get_betti_lap(pointcloud,distances)

np.save("b0_cb7.npy",b0)
np.save("b1_cb7.npy",b1)
np.save("l0_cb7.npy",l0)
np.save("l1_cb7.npy",l1)











