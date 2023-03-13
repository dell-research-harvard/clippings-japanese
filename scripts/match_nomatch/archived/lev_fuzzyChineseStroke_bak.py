# %%
# Import dependencies 
import time
import json
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import pickle
from hyperopt import hp
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
from itertools import repeat
from fuzzychinese import FuzzyChineseMatch
import matplotlib.pyplot as plt
import textdistance as td

# %%
def list_lev(word,list2):
    dist_list = []
    smallest_dist = 1000
    for word2 in tqdm(list2):
        if abs(len(str(word2[0]))-len(str(word))) > smallest_dist:
            dist = 1000
        dist = td.levenshtein(str(word2[0]),str(word))#Change it to our defined edit distance
        #print(dist)
        dist_list.append(dist)
        if dist<smallest_dist:
            smallest_dist = dist
    min_dist = float(np.min(dist_list))
    min_dist_word_path = list2[np.argmin(dist_list)] # Which word in the ground truth dict get matched to

    return [min_dist, min_dist_word_path]

def list_fuzzyChinese(raw_word,test_dict, title_dict, task_name):# raw_word is the partner list/list 1, test_dict is the  title list/list 2
    #return all the results, just pass in two lists, don't pass in too many other things.
    '''
    Input: test_dict, raw_word
    Output: nearest neighbor word list, nearest neighbor dist list  
    '''
    fcm = FuzzyChineseMatch(ngram_range=(3,3),analyzer="stroke")
    fcm.fit(test_dict)
    top1_similar_stroke = fcm.transform(raw_word,n=1) # This is the nearest neighbor list - return top 10 similar

    res = pd.concat([
        pd.DataFrame(top1_similar_stroke,columns=[f'fuzzychinese_stroke_{task_name}_matched_word_1']),
        pd.DataFrame(fcm.get_similarity_score(),columns=[f'fuzzychinese_stroke_{task_name}_word_dist_1']),
    ],axis = 1)

    res[f"fuzzychinese_char_{task_name}_matched_path_1"]=res.apply(lambda x:title_dict[x[f'fuzzychinese_char_{task_name}_matched_word_1']],axis=1)
    res[f"fuzzychinese_stroke_{task_name}_matched_path_1"]=res.apply(lambda x:title_dict[x[f'fuzzychinese_stroke_{task_name}_matched_word_1']],axis=1)
    return res
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partner_list", type=str, #/mnt/data01/yxm/homo/homo_match_dataset/japan
        default="/mnt/data01/yxm/record_linkage_clean_dataset/ocr_json/partner_list_clean_final_eff_gcv_paddle_easy.csv",# let's use csv 
        help="Path to Partners list")
    parser.add_argument("--json_path", type=str,
        default='/mnt/data01/yxm/record_linkage_clean_dataset/ocr_json')# This is the path to things on Guppy
    parser.add_argument("--match_task", type=list, # You can change the task as you wantf the gcv is already gcv
        default=[# TK task# Please update the GCV titles list - also make to no dup...
        ['gcv_2_gcvtk','gcv_ocr_partner','gcv_tk_title_nodup_0305_final.json','cjk'], 
        ['eff_2_efftk','effocr_partner','effocr_tk_title_nodup_0305_final.json','eff'],],
        help="Matching tasks to perform")
    parser.add_argument("--lev", action="store_true", default=False, 
        help="Levenstein Distance Matching")
    parser.add_argument("--fuzzychinese", action="store_true", default=False, 
        help="Fuzzychinese Stroke Matching")
    # Save output!
    parser.add_argument("--save_output", type=str, required=True, 
        help="Save output!")
    args = parser.parse_args()
    os.makedirs(args.eval_accuracy_output,exist_ok=True)

    partner_list = pd.read_csv(args.partner_list)
    match_task = args.match_task
    # Save output
    os.makedirs(args.save_output, exist_ok=True)
    partner_dict_list = []
    # Iterate over dfferent tasks
    '''
    list 1 is the source
    list 2 is what we parallize on since it is vert long
    '''
    store_time = {}
    # Run the matching if do_match is True
    for task_name, partner_ocr_choice, title, homo_type in tqdm(match_task):
        with open(os.path.join(args.json_path,f'{title}')) as f:
            title_list = json.load(f) # This is list2: title list
        partner_list_for_match = partner_list[partner_ocr_choice].values.tolist()
        # Maybe you can change the format of this dataset...
        if args.lev:
            start_time = time.time()
            with Pool(32) as p:
                mindist_WordPath_list = p.map(partial(list_lev,list2=title_list),partner_list_for_match) #result_list is the picture
            time_span = time.time()-start_time
            store_time[f"{task_name}_lev"] = time_span
            with open(os.path.join(args.save_output,'time_speed.json'),'w') as f:
                json.dump(store_time, f, ensure_ascii=False)

            matched_list, distance_list, path_list = map(lambda x: list(x),repeat([],3))
            for id, (list1_ele, word_dist_min) in enumerate(zip(partner_list,mindist_WordPath_list)):
                distance_list.append(round(word_dist_min[0],2))
                matched_list.append(word_dist_min[1][0])
                path_list.append(word_dist_min[1][1])

            df_match_result = pd.DataFrame({f'lev_{task_name}_matched_word_1':matched_list, \
                f'lev_{task_name}_matched_word_dist_1':distance_list, \
                f'lev_{task_name}_matched_path_1': path_list})

            df_matched = pd.concat([df_match_result,partner_list], axis=1)
            df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{task_name}_lev.csv'))

        if args.fuzzychinese:
            df_matched = pd.read_csv(os.path.join(args.save_output,f'df_full_matched_{task_name}_lev.csv'))
            raw_word = partner_list_for_match # This is the for matching
            title_dict = {}
            for_test = []
            for title in title_list:
                title_dict[title[0]] = title[1]
                for_test.append(title[0])

            test_dict = pd.Series(for_test) # The two lists are in the same length 
            # Divide it into several epochs...
            all_dict_list = []
            start_time = time.time()
            for i in range(0,len(raw_word),5000): # have to chunk it otherwise it get too big
                res = list_fuzzyChinese(raw_word[i:min(i+5000,len(raw_word))],test_dict,title_dict,task_name)
                res.to_csv(os.path.join(args.save_output,f'check_fuzzy_{i}.csv'))
                all_dict_list.append(res)

            time_span = time.time()-start_time
            store_time[f"{task_name}_fuzzychinese"] = time_span
            with open(os.path.join(args.save_output,'time_speed.json'),'w') as f:
                json.dump(store_time, f, ensure_ascii=False)

            for id, res in enumerate(all_dict_list):
                if id == 0:
                    df_matched = res
                else:
                    df_matched = pd.concat([df_matched,res])
            df_matched.to_csv(os.path.join(args.save_output,f'matched.csv'))
            df_matched.reset_index(drop = True,inplace = True)
            df_matched = pd.concat([df_matched,partner_list],axis = 1)
            df_matched.to_csv(os.path.join(args.save_output,f'df_full_matched_{task_name}_fuzzychinese.csv'))
