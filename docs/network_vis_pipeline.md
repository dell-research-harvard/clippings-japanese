For replicating Figure 4 of the paper , the following is the pipeline:

In the following, G1 is the graph for 'Clean OCR Multimodal CLIPPINGS' and G2 is the graph for 'Noisy OCR Levenshtein.' 

Step1: run 'Preparing_Network_G1.py' (You need to check whether the path is right). 
       -It needs the output df of infer_clippings (args : im_wt 0.5 as input
       -The output will be 'G1.pickle' which stores the graph G1. 
Step2: run 'Creating_Graph_Layout_G1.py' (It will take 10 to 30 minutes)
       -It needs 'G1.pickle'as input.
       -The output will be 'pos_G1.pickle' whichs stores the layout of G1 for visualization.
Step3: run 'Preparing_Network_G2.py'
       -It needs 'gcv_2_gcvtk_lev_1_31861.csv' as input.
       -The output will be 'G2.pickle' which stores the graph G2. 
Step4  run  'Creating_Graph_Layout_G2.py'
       -It needs 'G1.pickle', 'pos_G1.picke', and 'G2.pickle'as inputs to fix layout positions of nodes in G2 that appear in G1 as well. 
       -The output will be 'pos_G2.pickle'
Step5  run 'G1_Plt.py'
       -It needs 'G1.pickle' and 'pos_G1.pickle' as inputs.
       -The output will be 'Clean OCR Multimodal CLIPPINGS.html'
Step6  run 'G2_Plt.py'
       -It needs 'G2.pickle' and 'pos_G2.pickle' as inputs.
       -The output will be 'Noisy OCR Levenshtein.html'