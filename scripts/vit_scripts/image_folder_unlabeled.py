####Prepare crops for fine-tuning - make an image folder with the images

import pandas as pd
import numpy as np
import os
import json
from glob import glob
import random

def get_base_name(path):
    return os.path.basename(path)

def make_path_list(split_json,root_folder_path):
    with open(split_json) as f:
        data = json.load(f)
        data=data["images"]
    path_list = []
    for i in range(len(data)):
        path_list.append(os.path.join(root_folder_path,data[i]["file_name"]))
    return path_list







##Run as script
if __name__ == '__main__':

    datasets_to_choose="both"
    add_images=True
    random_add=True
    random_add_train=False

    pr_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_v2/PR_matched.csv')
    tk_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_v2/TK_matched.csv')

    ##Source and target columns contain paths to the image
    ##Concat the two dataframes and drop duplicates

    if datasets_to_choose=="PR":
        stacked_df = pr_df
    elif datasets_to_choose=="TK":
        stacked_df = tk_df
    else:
        stacked_df = pd.concat([pr_df,tk_df],axis=0)

    
    source_files=stacked_df["source"]
    all_files=stacked_df["source"] + stacked_df["target"]
    


    ###Save path
    save_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/jp_tk_unlabeled/images/'
    

    ##Rm the save path if it exists
    if os.path.exists(save_path):
        os.system("rm -r {}".format(save_path))
    


    ##MAke the save path if it doesn't exist recursively
    if not os.path.exists(save_path):
        os.makedirs(save_path)




    tk_all_files=glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/tktitle_image_v5_0105_v3/*")
    # if random_add==True:
    #     ##Sample n images from the tk_all_files
    #     tk_all_files=random.sample(tk_all_files,5000)

    # else :
    #     ##Load selected subset of images
    #     with open("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_10k_v2/selected_tk_images.json","r") as f:
    #         tk_all_files=json.load(f)

    for i in range(len(tk_all_files)):
        if tk_all_files[i] in all_files:
            continue
        tk_file_name=get_base_name(tk_all_files[i])
        tk_file_folder = os.path.join(save_path,tk_file_name.split(".")[0])
        if not os.path.exists(tk_file_folder):
            os.mkdir(tk_file_folder)
        ##Copy the image to the folder
        os.system("cp {} {}".format(tk_all_files[i],tk_file_folder))


        





    ###Make a df with counts of files in each folder
    folder_list = os.listdir(save_path)

    folder_count_df = pd.DataFrame(columns=['folder_name','count'])
    for folder in folder_list:
        folder_path = os.path.join(save_path,folder)
        count = len(os.listdir(folder_path))
        folder_count_df = folder_count_df.append({'folder_name':folder,'count':count},ignore_index=True)

    print(folder_count_df.head(100))



    ###Now split the dataset
    ##Split it such that all of source image folders are covered in train, test and val in the right prop
    ##Add empty folders of the tk_all_files to the root folder used in fine-tuning

    fine_tuning_root="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_v3/"

    for i in range(len(tk_all_files)):
        if tk_all_files[i] in all_files:
            continue
        tk_file_name=get_base_name(tk_all_files[i])
        tk_file_folder = os.path.join(fine_tuning_root,tk_file_name.split(".")[0])
        if not os.path.exists(tk_file_folder):
            os.mkdir(tk_file_folder)
        

    