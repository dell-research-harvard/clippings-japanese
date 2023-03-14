import numpy as np
import pandas as pd
import os
import random
import pickle
from copy import deepcopy
import networkx







if __name__ == '__main__':
    #The G1.picklefile is created by the script: Preparing_Network_G1.py
    with open('G1.pickle', 'rb') as handle:
        G1 = pickle.load(handle)
    #Convert G to a directed graph
    G1 = networkx.DiGraph(G1)

    #The G2.picklefile is created by the script: Preparing_Network_G2.py
    with open('G2.pickle', 'rb') as handle:
        G2 = pickle.load(handle)
    #Convert G to a directed graph
    G2 = networkx.DiGraph(G2)

    #The pos_G1.picklefile is created by the script: Creating_Graph_Layout_G1.py
    with open('pos_G1.pickle', 'rb') as handle:
        pos_G1 = pickle.load(handle)
    
    #Create a list of nodes in G1
    nodes_list_1 = list(G1.nodes())
 
    #The code will run from 20 to 40 minutes
    ##Spring_layout uses g Fruchterman-Reingold force-directed algorithm. 
    # The algorithm simulates a force-directed representation of the network treating edges as springs holding nodes close, 
    # while treating nodes as repelling objects, sometimes called an anti-gravity force. 
    # Simulation continues until the positions are close to an equilibrium.
    # See the documentation of networkx for more details
    # 42 is a arbitrary seed, please choose yours
    pos_G2 = networkx.spring_layout(G2,fixed=nodes_list_1,pos=pos_G1,seed=42)
    with open('pos_G2.pickle', 'wb') as fb:
        pickle.dump(pos_G2, fb)

    




