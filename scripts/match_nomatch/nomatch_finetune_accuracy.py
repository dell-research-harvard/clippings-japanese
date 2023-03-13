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
        return row[image_id] if row[dist]<thres else -9
    else:
        return row[image_id] if row[dist]>thres else -9

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
# %%
# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_results_dir", type=str, default='./dataset/full_inference', help='the dir contains files you want to calculate matched accuracy for')
    parser.add_argument("--match_ground_truth", type=str, default='./dataset/ground_truth/truth_TK_partnerpath_2_titlepath_0308.json', help='path to ground truth file')
    parser.add_argument("--nomatch_val_subset", type=str, default='./dataset/nomatch_split/val_list_0313_v1.json')
    parser.add_argument("--nomatch_test_subset", type=str, default='./dataset/nomatch_split/test_list_0313_v1.json')
    parser.add_argument("--output_dir", type=str, default='./nomatch_output')
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--best_nomatch_threshold", type=str, default='./dataset/nomatch_split/nomatch_accuracy_top1_0313.json')# The second element is the threshold

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

        matched_results["TK_truth_image"] = matched_results.apply(lambda x:truth_TK_partnerpath_2_titlepath[x["source"]] if x["source"] in truth_TK_partnerpath_2_titlepath else [-9], axis=1)
        # Clean this to a list
        df_val = matched_results[matched_results["source"].isin(nomatch_val_subset)]
        df_test = matched_results[matched_results["source"].isin(nomatch_test_subset)]
        
        print('nomatch validation set size',len(df_val),'nomatch test set size',len(df_test))
        # The function
        if args.finetune:
            accuracy_dict[file_name] = match_result(file_name)
            with open(os.path.join(args.output_dir,'nomatch_accuracy_finetune.json'),'w') as f:
                json.dump(accuracy_dict, f, ensure_ascii=False)

        else:
            
            with open(args.best_nomatch_threshold) as f:best_nomatch = json.load(f)
            best_no_match_thresh = best_nomatch[file_name][0]
            if 'lev' in file_name:
                df_val['prediction']=df_val.apply(lambda row:label_predict(row,best_no_match_thresh,"distance","matched_tk_path",1),axis=1)
            else:
                df_val['prediction']=df_val.apply(lambda row:label_predict(row,best_no_match_thresh,"distance","matched_tk_path",0),axis=1)

            df_val["accuracy"]=df_val.apply(lambda x: 1 if x["prediction"] in x["TK_truth_image"] else 0,axis=1)
            val_accuracy=df_val["accuracy"].mean()
            df_val.to_csv(os.path.join(args.output_dir, f'val_{file_name}'))

            #print(random_best)
            if 'lev' in file_name:
                df_test['prediction']=df_test.apply(lambda row:label_predict(row,best_no_match_thresh,"distance","matched_tk_path",1),axis=1)
            else:
                df_test['prediction']=df_test.apply(lambda row:label_predict(row,best_no_match_thresh,"distance","matched_tk_path",0),axis=1)

            df_test["accuracy"]=df_test.apply(lambda x: 1 if x["prediction"] in x["TK_truth_image"] else 0,axis=1)
            test_accuracy=df_test["accuracy"].mean()
            df_test.to_csv(os.path.join(args.output_dir, f'test_{file_name}'))

            accuracy_dict[file_name] = {}
            accuracy_dict[file_name][best_no_match_thresh] = [val_accuracy, test_accuracy] 
        
            with open(os.path.join(args.output_dir,f'nomatch_accuracy_best.json'),'w') as f:
                json.dump(accuracy_dict, f, ensure_ascii=False)

