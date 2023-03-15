import numpy as np
import pandas as pd
import os
import random
import pickle
import networkx
import matplotlib.pyplot as plt
from bokeh.io import  show,save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet,Text, BoxSelectTool,Legend, LegendItem,LinearColorMapper,ColorBar,BasicTicker,LogColorMapper
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.io import output_notebook, show, save
from bokeh.palettes import Reds256
from bokeh.transform import linear_cmap

if __name__ == '__main__':


    #Import G2 and pos_G2
    with open('G2.pickle', 'rb') as handle1:
        G2 = pickle.load(handle1)
    with open('pos_G2.pickle', 'rb') as handle2:
        pos_G2 = pickle.load(handle2)
    #Make G  a directed graph
    G2 = networkx.DiGraph(G2)

    # For visualization, we adjust the node size in the plot according to node's degree
    # The key idea is to make (1) too small degree nodes less visable and (2) medium and larger degree nodes visible in the plot
    # The prepared G1 has already haved an attribute called 'degree' for each node
    # If degree < 16, then set adjusted_node_size to 0.1*(degree), otherwise set adjusted_node_size to degree
    adjusted_node_size = dict([(node, 0.2*np.log(degree)) if degree < 16 else (node, 0.14*degree) for node, degree in networkx.degree(G2)])
    networkx.set_node_attributes(G2, name='adjusted_node_size', values=adjusted_node_size)


    #----Create a variabel based on Avg_Top3_Zaibatsu_Distance for shading----
    #For a large porportion of observaion, Avg_Top3_Zaibatsu_Distance is below 3
    # So for coloring, we rescale the value >=3 to be more "aggregated" for color shading
    Avg_Top3_Zaibatsu_Distance_d = networkx.get_node_attributes(G2, 'Avg_Top3_Zaibatsu_Distance')


    #Replace the value of 999 with 15
    Avg_Top3_Zaibatsu_Distance_d = {k: 15 if v == 999 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    #Replace the value of 3,4 with 3
    Avg_Top3_Zaibatsu_Distance_d = {k: 3 if v == 4 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    Avg_Top3_Zaibatsu_Distance_d = {k: 3 if v == 3 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    #Replace the value of 5,6 with 4
    Avg_Top3_Zaibatsu_Distance_d = {k: 4 if v == 5 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    Avg_Top3_Zaibatsu_Distance_d = {k: 4 if v == 6 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    #Replace the value of 7,8 with 5
    Avg_Top3_Zaibatsu_Distance_d = {k: 5 if v == 7 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    Avg_Top3_Zaibatsu_Distance_d = {k: 5 if v == 8 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    #Replace the value > 8 but < 15 with 6
    Avg_Top3_Zaibatsu_Distance_d = {k: 6 if v > 8 and v < 15 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}
    #Replace the value of 15 with 7
    Avg_Top3_Zaibatsu_Distance_d = {k: 7 if v == 15 else v for k, v in Avg_Top3_Zaibatsu_Distance_d.items()}




    #Add the dictionary to the network attribute "Relative_closeness_centrality"
    networkx.set_node_attributes(G2, Avg_Top3_Zaibatsu_Distance_d, 'Avg_Top3_Zaibatsu_Distance_d')



    #Set a random seed
    np.random.seed(42)
    #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
    size_by_this_attribute = 'adjusted_node_size'
    #Create a dictionary map each node and 'Top3_Zaibatsu_Distance'
    Avg_Top3_Zaibatsu_Distance_d= networkx.get_node_attributes(G2, 'Avg_Top3_Zaibatsu_Distance_d')
    color_by_this_attribute = 'Avg_Top3_Zaibatsu_Distance_d'

    #Choose a title!
    title = 'Noisy OCR Levenshtein'
    #Create a list named color palette, and contain 11 elements as 'NA'
    #color_palette = ['NA' for i in range(5)]
    #Filling colors

    #------Define the color palette------
    # We first create a color_list from Reds256 and adjust Reds256 to make it more shading through the average distance to top3 firms in Japan 

    color_list=Reds256
    #Convert the tuple into a list
    color_list= list(color_list)

    # Convert the list into a tuple
    #Replace the 70 to 100 elements of color_palette with light red but darker than #ff9d9d
    color_list[70:100] = ['#f76565' for i in range(30)]
    #Replace the 100 to 130 elements of color_palette with light red
    color_list[100:130] = ['#ff9d9d' for i in range(30)]
    #Replace the 130 to 150 elements of color_palette with even lighter red
    color_list[130:150] = ['#ffc5c5' for i in range(20)]
    #Replace the 150to 170 elements of color_palette with even lighter red
    color_list[150:170] = ['#ffebeb' for i in range(20)]
    #Replace the 170 to 190 elements of color_palette with even lighter red
    color_list[170:190] = ['#ffecec' for i in range(20)]
    #Replace the 190 to the last elements of color_palette with even lighter red
    color_list[190:] = ['#ffecec' for i in range(66)]

    # Replace the last 25 elements of color_palette with gray
    color_list[-25:] = ['#5A5A5A' for i in range(25)]


    color_palette = tuple(color_list)


    # The same script here can be used to generate an iteractive plot

    #Establish which categories will appear when hovering over each node 
    HOVER_TOOLTIPS = [
        ("Company", "@index"),
            ("Degree", "@degree")
            
    ]

    #Create a network plot using Bokeh
    #We zoom into [-.5,.5] x [-.5,.5] to focus on the center of the network
    plot = figure(title=title, x_range=Range1d(-0.5,0.5), y_range=Range1d(-0.5,0.5), tooltips=HOVER_TOOLTIPS, plot_width=700, plot_height=700)
    #Create a network graph object using networkx 
    graph = from_networkx(G2, pos_G2, scale=1, center=(0,0))
    #Set node sizes and colors according to node degree (color as spectrum of color palette)
    minimum_value_color = min(graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(graph.node_renderer.data_source.data[color_by_this_attribute])
    #Set the node size 
    #We adjust line_alpha to make the boundary of nodes less opaque
    graph.node_renderer.glyph = Circle(size=size_by_this_attribute, line_alpha=0.08, fill_color=linear_cmap(color_by_this_attribute, color_palette, minimum_value_color, maximum_value_color))
    #Set the edge opacity
    #We set line_alpha to 0 to make the edges invisible in case the whole plot become a hairball
    graph.edge_renderer.glyph = MultiLine(line_color='gray',line_alpha=0, line_width=0)

    #---Set the legen---
    #Set the legend tick and labels
    #The labels are set manually to make the labels more readable in the plot

    tick_labels={0: "0", 0.5:"" ,1: "0", 1.5:"", 2: "", 2.5: "", 3:"3", 3.5:"", 4: "", 4.5: "", 5: "", 5.5: ">6", 6: "", 6.5: "", 7:"",7.5: "Not Connected"}
    color_mapper = LinearColorMapper(palette=color_palette, low=minimum_value_color, high=maximum_value_color)
    color_bar = ColorBar(color_mapper=color_mapper, 
                        label_standoff=10,
                        border_line_color='gray',
                        border_line_dash='dashed', 
                        location='top_left',
                        width=10, 
                        height=180,
                        major_tick_line_width=0,
                        title='    Average Distance to Top3',
                        title_text_font_style='bold',
                        major_label_overrides = tick_labels,
                        ticker=BasicTicker(desired_num_ticks=11),
                        major_label_text_font_size='7pt',
                        title_standoff = 3)


    ##Hide the axis ticks and axis labels
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None
    plot.xaxis.major_label_text_color = None
    plot.yaxis.major_label_text_color = None


    plot.add_layout(color_bar)



    plot.renderers.append(graph)

    print('Exporting Noisy OCR Levenshtein.html.....')

    #You can export the plot as a .png through the bokeh plot .html
    show(plot)
    save(plot, filename=f"{title}.html")




