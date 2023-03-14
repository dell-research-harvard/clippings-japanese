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
        G = pickle.load(handle)
    #Convert G to a directed graph
    G = networkx.DiGraph(G)
 
    #The code will run from 20 to 40 minutes
    ##Spring_layout uses g Fruchterman-Reingold force-directed algorithm. 
    # The algorithm simulates a force-directed representation of the network treating edges as springs holding nodes close, 
    # while treating nodes as repelling objects, sometimes called an anti-gravity force. 
    # Simulation continues until the positions are close to an equilibrium.
    # See the documentation of networkx for more details
    # 42 is a arbitrary seed, please choose yours
    pos_G1 = networkx.spring_layout(G,seed=42)
    with open('pos_G1.pickle', 'wb') as fb:
        pickle.dump(pos_G1, fb)

    




