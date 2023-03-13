# %%
# Import dependencies 
import json
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
from pathlib import Path

# %%
def same_matched(a,b):
    '''Whether the matched is in ground truth'''
    for ele in a:
        ele_name = Path(ele).name
        for truth in b:
            if ele_name == Path(truth).name: # If any of the ele in a equals b, return 1, after the iter, if nothing returns, just return 0
                return 1
    return 0
# %%
# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_results_dir", type=str, default='./dataset/full_inference', help='the dir contains files you want to calculate matched accuracy for')
    parser.add_argument("--match_ground_truth", type=str, default='./dataset/ground_truth/truth_TK_partnerpath_2_titlepath_0308.json', help='path to ground truth file')
    parser.add_argument("--match_test_subset", type=str, default='./dataset/match_split/test_paths.json')
    parser.add_argument("--source_image_path", type=str, default='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/pr_partner_crop_36673/')
    parser.add_argument("--output_dir", type=str, default="./match_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok = True)
    accuracy_dict = {}

    with open(args.match_ground_truth) as f:
        truth_TK_partnerpath_2_titlepath = json.load(f)

    with open(args.match_test_subset) as f:
        matched_test_subset_load = json.load(f)
    # Should only use file names
    matched_test_subset = []
    for image in matched_test_subset_load["images"]:
        image_name = args.source_image_path + image["file_name"].split('/')[-1]+'.png'
        matched_test_subset.append(image_name)

    matched_results_csv_list = glob(os.path.join(args.match_results_dir,"*.csv"))

    for matched_results_csv in matched_results_csv_list:
        file_name = matched_results_csv.split('/')[-1]
        matched_results = pd.read_csv(matched_results_csv)

        matched_results["TK_truth_image"] = matched_results.apply(lambda x:truth_TK_partnerpath_2_titlepath[x["source"]] if x["source"] in truth_TK_partnerpath_2_titlepath else None,axis=1)
        matched_results.dropna(subset=["TK_truth_image"],inplace = True,axis = 0)
        matched_results_in_test = matched_results[matched_results['source'].isin(matched_test_subset)]           
        matched_accuracy = {}

        matched_results_in_test[f"accuracy"] = matched_results_in_test.apply(lambda x:same_matched([x["matched_tk_path"]],x["TK_truth_image"]), axis = 1)
        accuracy = matched_results_in_test[f"accuracy"].mean()

        accuracy_dict[file_name] = accuracy
        matched_results_in_test.to_csv(os.path.join(args.output_dir,f'{file_name}_with_accuracy.csv'))

        with open(os.path.join(os.path.join(args.output_dir,'matched_accuracy.json')),'w') as f:
            json.dump(accuracy_dict,f,ensure_ascii=False)

# %%