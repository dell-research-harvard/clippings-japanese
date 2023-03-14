import numpy as np
import pandas as pd
import os
import random
import pickle
from copy import deepcopy
import networkx


def preprocess_df(df, *args):
    df = df[list(args)]
    df.replace(-9, np.nan, inplace=True)
    return df


if __name__ == '__main__':



    #Path = the file "matched_ab_filtered.xlsx" in the same folder as this script
    path = os.path.join(os.getcwd(), 'mean_norm_1_effocr_partner_tk_match_matched_33408.csv')
    df = pd.read_csv(path)

    df=preprocess_df(df,'partner_main_title','partner_type','matched_tk_text')

    ####Create Directed Graph with "Forward Linkage" of the Supply Chain
    ###Step: 1 - Create a new dataframe that has "forward direction""
    ###Step: 2 - Create a graph from the new dataframe
    ###Step: 3 - Create a network plot from the graph
    ###The following scripts ndeed modifies the df_list so there is ‘SettingWithCopyWarning’  message

    #Drop all rows with NaN values
    df_list = df.dropna()
    #Create new colums named "source" with the same value as "partner_main_title" if the value in "partner_type" is "fanmai", otherwise, the value is "partner_text"
    df_list['source'] = df.apply(lambda x: x['partner_main_title'] if x['partner_type'] == 'fanmai' else x['matched_tk_text'], axis=1)
    #Create new colums named ""named "target" with the same value as "partner_text" if the value in "partner_type_list" is not "fanmai", otherwise, the value is "partner_main_title"
    df_list['target'] = df.apply(lambda x: x['matched_tk_text'] if x['partner_type'] == 'fanmai' else x['partner_main_title'], axis=1)

    #Create a new dataframe with only the columns we need including source and target
    df_list = df_list[['source','target']]

    #Keep only the rows where the source and target are not the same
    df_list = df_list[df_list['source'] != df_list['target']]

    G = networkx.from_pandas_edgelist(df_list, source='source', target='target',  create_using=networkx.DiGraph())

    ##Adding degree centrality of the nodes
    degrees = dict(networkx.degree(G))
    networkx.set_node_attributes(G, name='degree', values=degrees)

 



    ##Adding Top3 firm list 

    Sumitomo_list = ['住友','大阪商船','日本電気','日新化工']
    Mitsubishi_list=['三菱','川崎重工','川崎製鐵','日本郵船','富士紡','大平鑛山','旭硝子']
    Mitsui_list=['三井','王子製紙','十條製紙','本州製紙','東京芝浦電気','東京電器','北芝電機','東芝','東京電燈器具','足立鋼業','東海爐材','西芝電機','鐘淵','北海道炭礦汽船','第一物產','日新通商','豐田']
    #Create a list collect all elements in the above lists
    Zaibatsu_keyword_list = Sumitomo_list + Mitsui_list + Mitsubishi_list
    #Create a node list of all the nodes in the graph G
    node_list = list(G.nodes())

    ##Add Top3 firms and other Zaiabtsu firms dummy as node attributes
    #Creat a variable called "Sumitomo" for node_list  if the node name includes the elements of Sumi_list then it eqauls 1, otherwise it equals 0
    Sumitomo = dict([(node, 1) if any(x in node for x in Sumitomo_list) else (node, 0) for node in node_list])
    #Creat a variable called "Mitsubishi" for node_list  if the node name includes the elements of Mits_list then it eqauls 1, otherwise it equals 0
    Mitsubishi = dict([(node, 1) if any(x in node for x in Mitsubishi_list) else (node, 0) for node in node_list])
    #Creat a variable called "Mitsui" for node_list  if the node name includes the elements of Mitsui_list then it eqauls 1, otherwise it equals 0
    Mitsui = dict([(node, 1) if any(x in node for x in Mitsui_list) else (node, 0) for node in node_list])
  
    #Set the attributes of the graph G
    networkx.set_node_attributes(G, name='Mitsubishi', values=Mitsubishi)
    networkx.set_node_attributes(G, name='Mitsui', values=Mitsui)
    networkx.set_node_attributes(G, name='Sumitomo', values=Sumitomo)
   

    #For convenience, we create a new variable called top3_zaibatsu to represent the top3 firms in Japan: 
    # Mitsubishi,Mitsui, and Sumitomo
    #if the node attribute Mitsubushi =1 then it equals 1
    # if the node attribute  Mitsui=1 then it eqauls 2
    # if the node attribute Sumitomo =1, then 3, 
    #otherwise it equals 0
    top3_zaibatsu = dict([(node, 1) if Mitsubishi[node] == 1 else (node, 2) if Mitsui[node] == 1 else (node, 3) if Sumitomo[node] == 1 else (node, 0) for node in node_list])
    networkx.set_node_attributes(G, name='top3_zaibatsu', values=top3_zaibatsu)




    ###Compute the shortest_path_length 
    #---------Warning: it may take a couple of minutes to run this part of the code--------
    #The shortest path length is the number of edges in the shortest path between two nodes.
    path_length= dict(networkx.all_pairs_shortest_path_length(G))

    ##Convert the path_length into the dataframe for further computation
    #Convert the path_length dictionary into a dataframe
    #---------Warning: it may take a couple of minutes to run this part of the code--------
    path_length_df = pd.DataFrame.from_dict(path_length)


    #Creae a list of all the nodes with top3_zaibatsu >0
    top3_zaibatsu_list = [node for node in node_list if top3_zaibatsu[node] > 0]

    #Create a list of all the nodes with top3_zaibatsu =0
    non_top3_zaibatsu_list = [node for node in node_list if top3_zaibatsu[node] == 0]



    #---Creat a input dataframe including firms to all zaibatsu/top3_zaibatsu---
    #If row names of path_length_df are not one of Zaibatsu_list, then drop the row
    #Rows: all Firms
    #Columns: all zaibatsu
    #Values in each cell: the shortest distance from row firm to column zaibatsu
    path_length_df_top3 = path_length_df.drop([x for x in path_length_df.index if x not in top3_zaibatsu_list], axis=0)
    #Transpose the path_length_df_sub for further computation
    path_length_df_top3 = path_length_df_top3.T




    #---Creating Average Supply Chain Distance to the Top3 firms in Japan---

    #--------Step 0: Prepare the df--------

    #Creat four input dataframe including (1) firms to all top3 (2) firms to non-top3 (3) firms to all Zaibatsu (4) firms to non-zaibatsu
    #If row names of path_length_df are not one of Zaibatsu_list, then drop the row
    #Rows: all Firms
    #Columns: all zaibatsu/ non-zaibatsu
    #Values in each cell: the shortest distance from row firm to column zaibatsu/non-zaibatsu
    #Value will be NA if the firm i correspond to same firm for column j 
    path_length_df_top3_cd = path_length_df.drop([x for x in path_length_df.index if x not in top3_zaibatsu_list], axis=0)
    path_length_df_non_top3_cd = path_length_df.drop([x for x in path_length_df.index if x in top3_zaibatsu_list], axis=0)
    #Transpose the path_length_dfs for convenience of computation
    path_length_df_top3_cd = path_length_df_top3_cd.T
    path_length_df_non_top3_cd = path_length_df_non_top3_cd.T

    #Replace the value 0 with NA for all four dfs
    path_length_df_top3_cd = path_length_df_top3_cd.replace(0, np.nan)
    path_length_df_non_top3_cd = path_length_df_non_top3_cd.replace(0, np.nan)


    ##Create a new variable called 'Average Top3_Zaibatsu_Distance', 'Average All_Zaibatsu_Distance', 'Average Non_Top3_Zaibatsu_Distance', 'Average Non_Zaibatsu_Distance' with value as the average value of all other variables
    path_length_df_top3_cd ['Avg_Top3_Zaibatsu_Distance'] = path_length_df_top3_cd .mean(axis=1)
    path_length_df_non_top3_cd ['Avg_Non_Top3_Zaibatsu_Distance'] = path_length_df_non_top3_cd .mean(axis=1)

    #Replace the value NaN with 999 for all four dfs
    path_length_df_top3_cd = path_length_df_top3_cd.replace(np.nan, 999)
    path_length_df_non_top3_cd = path_length_df_non_top3_cd.replace(np.nan,999)

    #Create a new node attribute called "Avg_Top3_Zaibatsu_Distance", "Avg_All_Zaibatsu_Distance" , "Avg_Non_Top3_Zaibatsu_Distance", "Avg_Non_Zaibatsu_Distance"  in G
    networkx.set_node_attributes(G, path_length_df_top3_cd['Avg_Top3_Zaibatsu_Distance'], 'Avg_Top3_Zaibatsu_Distance')
    networkx.set_node_attributes(G, path_length_df_non_top3_cd['Avg_Non_Top3_Zaibatsu_Distance'], 'Avg_Non_Top3_Zaibatsu_Distance')


    #Save G as a pickle file
    with open('G1.pickle', 'wb') as handle:
        pickle.dump(G, handle)