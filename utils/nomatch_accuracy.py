# %%
# Import dependencies 
import json
from glob import glob
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
import argparse
import numpy as np
from itertools import repeat

# Define functions
def label_predict(row,thres,dist,image_id,flag):# The prediction incorporates the no matches
    if flag==1:
        return row[image_id] if float(row[dist])<thres else -9
    else:
        return row[image_id] if float(row[dist])>thres else -9

def match_result(file_name): # The what should be named as task, which is better
    def objective_function(dist,image_id,flag,true):
        def objective(params):
            hp1=params['hp1']
            df_val['prediction']=df_val.apply(lambda row:label_predict(row,hp1,dist,image_id,flag),axis=1)
            df_val["accuracy"]=df_val.apply(lambda x: 1 if x["prediction"] in x[true] else 0,axis=1)
            accuracy_score=df_val["accuracy"].mean()
            negative_accuracy=-accuracy_score
            return negative_accuracy
        return objective

    space={'hp1':hp.uniform('hp1',0,8)}

    rand_trials = Trials()
    if 'lev' in file_name:
        best=fmin(fn=objective_function("distance","matched_tk_path",1,"TK_truth_image"),space=space,algo=rand.suggest,trials=rand_trials,max_evals=100)
    else:
        best=fmin(fn=objective_function("distance","matched_tk_path",0,"TK_truth_image"),space=space,algo=rand.suggest,trials=rand_trials,max_evals=100)

    random_best=best["hp1"]
    val_accuracy = -rand_trials.best_trial['result']['loss']

    if 'lev' in file_name:
        df_test['prediction']=df_test.apply(lambda row:label_predict(row,random_best,"distance","matched_tk_path",1),axis=1)
    else:
        df_test['prediction']=df_test.apply(lambda row:label_predict(row,random_best,"distance","matched_tk_path",0),axis=1)

    df_test["accuracy"]=df_test.apply(lambda x: 1 if x["prediction"] in x["TK_truth_image"] else 0,axis=1)
    test_accuracy = df_test["accuracy"].mean()
    return [random_best,val_accuracy,test_accuracy]

def calculate_nomatch_accuracy(matched_results = 'DATAFRAME', file_name = 'mean_norm_1_effocr_partner_tk_match.csv', levenshtein_match = False):
    with open('/path/to/data/record_linkage_clean_dataset/nomatch_thresh.json') as f:
        nomatch_thresh = json.load(f)
    with open("/path/to/data/record_linkage_clean_dataset/ground_truth/truth_TK_partnerpath_2_titlepath_0314.json") as f:
        truth_TK_partnerpath_2_titlepath = json.load(f)

    matched_results["TK_truth_image"] = matched_results.apply(lambda x:truth_TK_partnerpath_2_titlepath[x["source"]] if x["source"] in truth_TK_partnerpath_2_titlepath else [-9], axis=1)
    # Clean this to a list
    with open('/path/to/data/record_linkage_clean_code/new_subset/val_list_0314_v1.json') as f:
        nomatch_val_subset = json.load(f)

    with open('/path/to/data/record_linkage_clean_code/new_subset/test_list_0314_v1.json') as f:
        nomatch_test_subset = json.load(f)

    df_val = matched_results[matched_results["source"].isin(nomatch_val_subset)]
    df_test = matched_results[matched_results["source"].isin(nomatch_test_subset)]
    
    print('nomatch validation set size',len(df_val),'nomatch test set size',len(df_test))
    if levenshtein_match == True:
        df_test['prediction']=df_test.apply(lambda row:label_predict(row,float(nomatch_thresh[file_name][0]),"distance","matched_tk_path",1),axis=1)
    else:
        df_test['prediction']=df_test.apply(lambda row:label_predict(row,float(nomatch_thresh[file_name][0]),"distance","matched_tk_path",0),axis=1)

    df_test["accuracy"]=df_test.apply(lambda x: 1 if x["prediction"] in x["TK_truth_image"] else 0,axis=1)
    test_accuracy=df_test["accuracy"].mean()

    return test_accuracy
# %%
# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_results_dir", type=str, default='./dataset/full_inference', help='the dir contains files you want to calculate matched accuracy for')
    parser.add_argument("--match_ground_truth", type=str, default='./dataset/ground_truth/truth_TK_partnerpath_2_titlepath_0314.json', help='path to ground truth file')
    parser.add_argument("--nomatch_val_subset", type=str, default='./dataset/nomatch_split/val_list_0314_v1.json')
    parser.add_argument("--nomatch_test_subset", type=str, default='./dataset/nomatch_split/test_list_0314_v1.json')
    parser.add_argument("--output_dir", type=str, default='./nomatch_output')
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--best_nomatch_threshold", type=str, default='./dataset/nomatch_split/nomatch_acc.json')# The second element is the threshold

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.match_ground_truth) as f:truth_TK_partnerpath_2_titlepath = json.load(f)

    with open(args.nomatch_val_subset) as f:nomatch_val_subset = json.load(f)

    with open(args.nomatch_test_subset) as f:nomatch_test_subset = json.load(f)

    accuracy_dict = {}
    matched_results_csv_list = glob(os.path.join(args.match_results_dir,"*.csv"))

    for matched_results_csv in matched_results_csv_list:
        file_name = matched_results_csv.split('/')[-1]
        matched_results = pd.read_csv(matched_results_csv)

        # The function
        if args.finetune:
            if 'lev' in file_name:#Use random search for levenshtein match
                matched_results["TK_truth_image"] = matched_results.apply(lambda x:truth_TK_partnerpath_2_titlepath[x["source"]] if x["source"] in truth_TK_partnerpath_2_titlepath else [-9], axis=1)
                # Clean this to a list
                df_val = matched_results[matched_results["source"].isin(nomatch_val_subset)]
                df_test = matched_results[matched_results["source"].isin(nomatch_test_subset)]
                
                print('nomatch validation set size',len(df_val),'nomatch test set size',len(df_test))
                accuracy_dict[file_name] = match_result(file_name)
                with open(os.path.join(args.output_dir,'nomatch_accuracy_finetune.json'),'w') as f:
                    json.dump(accuracy_dict, f, ensure_ascii=False)
            else:
                thresh_list = [i/100 for i in range(0,101)]
                
                val_best = -9
                for thresh in thresh_list:
                    df_val['prediction']=df_val.apply(lambda row:label_predict(row,thresh,"distance","matched_tk_path",0),axis=1)
                    df_val["accuracy"]=df_val.apply(lambda x: 1 if x["prediction"] in x["TK_truth_image"] else 0,axis=1)
                    # Maybe also save the csv here for reference to errors picking
                    val_accuracy=df_val["accuracy"].mean()
                    df_test['prediction']=df_test.apply(lambda row:label_predict(row,thresh,"distance","matched_tk_path",0),axis=1)
                    df_test["accuracy"]=df_test.apply((lambda x: 1 if x["prediction"] in x["TK_truth_image"] else 0),axis=1)
                    test_accuracy=df_test["accuracy"].mean()
                    if val_accuracy>val_best:
                        val_best = val_accuracy
                        accuracy_dict[file_name] = [thresh, val_accuracy]
                    with open(os.path.join(args.output_dir,'nomatch_accuracy_finetune_grid_search.json'),'w') as f:
                        json.dump(accuracy_dict, f, ensure_ascii=False)         

        else:
            calculate_nomatch_accuracy(match_results = 'DATAFRAME', file_name = 'mean_norm_1_effocr_partner_tk_match.csv', levenshtein_match = False)
