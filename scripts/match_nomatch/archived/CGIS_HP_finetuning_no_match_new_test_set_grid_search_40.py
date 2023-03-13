# %%
'''
Input: 
1. homoglyph dicts
2. --match_task specifiy the task and the title dict
3. --json_path specify the path to the title dataset

Output:
1. All in args.save_output
2. Will be the csv files

Notes:
You may only want to perform simstring and homoglyph matching since levenshtein is taken care by StringDist in R, fuzzyChiense is not wanted

In most cases, the evaluation input directory will be the same as the save_output and should be run after the evaluation is done...
Actually, you should not use so many partners to run thing, you can only evaluate for the source that you have ground truth for
'''

# Import dependencies 
import json
from glob import escape, glob
import os
import pandas as pd
from tqdm import tqdm
import pickle
from hyperopt import hp
import matplotlib.pyplot as plt
import random
from hyperopt import rand, tpe
from hyperopt import fmin
from hyperopt import Trials

from multiprocessing import Pool
import faiss
import heapq
from functools import partial
import argparse
import numpy as np
from itertools import repeat
# from fuzzychinese import FuzzyChineseMatch
# from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
# from simstring.measure import CosineMeasure
# from simstring.measure import JaccardMeasure
# from simstring.measure import OverlapMeasure
# from simstring.measure import DiceMeasure
# from simstring.searcher import Searcher
# from simstring.database import DictDatabase
# from simstring.searcher import Searcher
from hyperopt import hp
from hyperopt import rand, tpe
import matplotlib.pyplot as plt

# %%
# load homoglyph dict, Eff for EffOCR only, CJK for Paddle/EasyOCR. Since this is very big, we don't want to pass it as an argument to function to slow down things
# print('loading homoglyph dict')
# with open("/mnt/data01/yxm/homo/Japan_match_hg_set/char_char_dist_dict_800_31_Hira_Kana_Bopo_Hans_Hant_Hang_Hani_Hrkt.pickle",'rb') as f:
#     cluster_dict_CJK = pickle.load(f)

# with open("/mnt/data01/yxm/homo/Japan_match_hg_set/char_char_dist_dict_800_31_EffOCR_chars.pickle",'rb') as f:
#  %%
# parse arguments
if __name__ == "__main__":
    
    # This is really quick to run
    SAVE_DIR = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_code/nomatch_tune_0313_grid_search_thresh'
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Load the partner list
    accuracy_dict = {}
    
    big_df_refined_cols_list = glob(os.path.join('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_code/full_inference',"*.csv"))

    for big_df_refined_cols_file in big_df_refined_cols_list:

        file_name = big_df_refined_cols_file.split('/')[-1]
        
        accuracy_dict[file_name] = {}
        big_df_refined_cols_for_nomatch = pd.read_csv(big_df_refined_cols_file)
        # When add ground truth data, Let's separately do for whether it is matched or not matched...
    # No need to finetune anything, just put the labels in, and only keep those matched
        with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ground_truth/truth_TK_partnerpath_2_titlepath_0308.json') as f:
            truth_TK_partnerpath_2_titlepath = json.load(f)

        def TK_label(path):
            if path in truth_TK_partnerpath_2_titlepath:
                return truth_TK_partnerpath_2_titlepath[path]
            else:
                return [-9]
        big_df_refined_cols_for_nomatch["TK_truth_image"] = big_df_refined_cols_for_nomatch.apply(lambda x:TK_label(x["source"]), axis=1)
        # The ground truth is already added in
        # Maybe you can first keep those that are in the nomatch set
        # Val
        with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_code/new_subset/val_list_0313_v1.json') as f:
            nomatch_val_subset = json.load(f)
        # Clean this to a list


        df_val = big_df_refined_cols_for_nomatch
        df_val = big_df_refined_cols_for_nomatch[big_df_refined_cols_for_nomatch["source"].isin(nomatch_val_subset)]
        # Change the NA in 
    
        # test set
        with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_code/new_subset/test_list_0313_v1.json') as f:
            nomatch_test_subset = json.load(f)
        

        # Clean this to a list

        df_test = big_df_refined_cols_for_nomatch[big_df_refined_cols_for_nomatch["source"].isin(nomatch_test_subset)]
        
        print('nomatch val',len(df_val),'nomatch test',len(df_test))
        # The function
        def label_predict(row,thres,dist,image_id,flag):# The prediction incorporates the no matches
            if flag==1:
                if row[dist]<thres:
                    return row[image_id]
                else:
                    return -9
            if flag==0:
                if row[dist]>thres:
                    return row[image_id]
                else:
                    return -9
        def same(a,b):
            if a in b:
                return 1
            else:
                return 0

        # Let's gen a list for iteration [0.5, 0.99]
        thresh_list = []

        x = 0
        for m in range(0,100):
            thresh_list.append(x+m*0.01)
        thresh_lev_list = [x+1 for x in thresh_list]

        for thresh, thresh_lev in zip(thresh_list, thresh_lev_list):
            if 'lev' in file_name:
                df_val['prediction']=df_val.apply(lambda row:label_predict(row,thresh_lev,"distance","matched_tk_path",1),axis=1)
            else:
                df_val['prediction']=df_val.apply(lambda row:label_predict(row,thresh,"distance","matched_tk_path",0),axis=1)
            df_val["accuracy"]=df_val.apply(lambda x: same(x["prediction"],x["TK_truth_image"]),axis=1)
            # Maybe also save the csv here for reference to errors picking
            val_accuracy=df_val["accuracy"].mean()
            df_val.to_csv(os.path.join(SAVE_DIR, f'val_{thresh}.csv'))

            #print(random_best)
            if 'lev' in file_name:
                df_test['prediction']=df_test.apply(lambda row:label_predict(row,thresh_lev,"distance","matched_tk_path",1),axis=1)
            else:
                df_test['prediction']=df_test.apply(lambda row:label_predict(row,thresh,"distance","matched_tk_path",0),axis=1)

            df_test["accuracy"]=df_test.apply(lambda x: same(x["prediction"],x["TK_truth_image"]),axis=1)
            test_accuracy=df_test["accuracy"].mean()
            df_test.to_csv(os.path.join(SAVE_DIR, f'test_{thresh}.csv'))

            #Inside of this store all the results when running!

            # This is something for storage
            accuracy_dict[file_name][thresh] = [val_accuracy, test_accuracy] 
        
            with open(os.path.join(SAVE_DIR,f'nomatch_accuracy_top1_change_{thresh}.json'),'w') as f:
                json.dump(accuracy_dict, f, ensure_ascii=False)
