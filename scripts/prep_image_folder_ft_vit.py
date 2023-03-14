####Prepare crops for fine-tuning - make an image folder with the images

import pandas as pd
import numpy as np
import os
import json
from glob import glob
import random

def get_base_name(path):
    return os.path.basename(path)


##Run as script
if __name__ == '__main__':

    pr_data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_partner.json"
    with open(pr_data) as f:
            pr_data = json.load(f)
    pr_partner_paths=[path for path in pr_data.keys()]
    pr_partner_text=[pr_data[path] for path in pr_data.keys()]

    ###OCR df
    partner_df_ocr=pd.DataFrame({"source":pr_partner_paths,"source_ocr_text":pr_partner_text})




    datasets_to_choose="both"
    # pr_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_v2/PR_matched.csv')
    # tk_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_v2/TK_matched.csv')
    pr_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/PR_matched_1092_appended.csv')

    ##Merge ocr df
    
    ##More labels
    # pr_more_df=pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/PR_match_more.csv')

    ###concat the two dfs
    # pr_df = pd.concat([pr_df,pr_more_df],axis=0)
    pr_df = pd.merge(pr_df,partner_df_ocr,on='source',how='left')

    tk_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/TK_matched_1207_appended.csv')
    ##Merge ocr df
    # tk_more_df=pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/TK_match_more.csv')
    # tk_df = pd.concat([tk_df,tk_more_df],axis=0)
    
    tk_df = pd.merge(tk_df,partner_df_ocr,on='source',how='left')
    ###Drop duplicates by source and source_ocr_text

    ###Get all source paths

    ###convert the 


    ###Add the ocr text to the dataframe

    ##Drop ocr text


    ###Add other variants
    # ##Other variant folder
    # other_var_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/source_target/training_data/*"
    # other_var_list=glob(other_var_folder) 

    # ###Load all df in the folder
    # other_df_list=[pd.read_csv(os.path.join(other_var_folder,path)) for path in other_var_list]

    # ##Concat all the other variants
    # other_variant_df=pd.concat(other_df_list,axis=0)

    ##Source and target columns contain paths to the image
    ##Concat the two dataframes and drop duplicates

    if datasets_to_choose=="PR":
        stacked_df = pr_df
    elif datasets_to_choose=="TK":
        stacked_df = tk_df
    else:
        stacked_df = pd.concat([pr_df,tk_df],axis=0)

    # #Concat other variants
    # stacked_df = pd.concat([stacked_df,other_variant_df],axis=0)
    
    remove_dup=True
    if remove_dup:
        ###Get df with targets + source text
        stacked_df_targets = stacked_df[['target','source_ocr_text']]

        ##Get df with source + source text
        stacked_df_sources = stacked_df[['source','source_ocr_text']]
        stacked_df_sources_copy = stacked_df[['source','source_ocr_text']]
        stacked_df_sources_copy.columns = ['target','source_ocr_text']

        ###GET targets from the sources as well! Same ocr text for different sources can give more targets for each source
        ###Merge stacked_df_sources to itself
        stacked_df_sources_cross = pd.merge(stacked_df_sources,stacked_df_sources_copy,on='source_ocr_text',how='left')

        ##SAve the csv

        stacked_df_sources_cross.to_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/stacked_df_sources_cross.csv',index=False)

        ##Merge the two dfs
        stacked_df = pd.merge(stacked_df_targets,stacked_df_sources,on='source_ocr_text',how='left')

        ###Concat the cross df
        stacked_df = pd.concat([stacked_df,stacked_df_sources_cross],axis=0)

        ##Sort by source_ocr_text,   source and then target
        stacked_df = stacked_df.sort_values(by=['source_ocr_text','source','target'])




        ##Each source now has the same list of targets if have the same source text
        ##We want only one source for each source text
        ##Sort by source_ocr_text, target and then source
        stacked_df = stacked_df.sort_values(by=['source_ocr_text','source','target'])
        stacked_df = stacked_df.drop_duplicates(subset=['source_ocr_text','target'])



        # ##Save the df
        stacked_df.to_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/PR_TK_stacked_df.csv',index=False,encoding='utf-8-sig')




    ###Print unique sources
    print("Unique sources: ",len(stacked_df['source'].unique()))





    ###Save path
    # save_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_synth/images/'
    save_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/images/'
    
    ##Rm the save path if it exists
    if os.path.exists(save_path):
        os.system("rm -r {}".format(save_path))
    

    ##MAke the save path if it doesn't exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ###Iterate through each row of stacked df , Make a folder for each source image and 
    # ## copy it in the folder along with the target image 
    ###Make a list of unique source images
    source_list = stacked_df['source'].unique()
    print("Number of source images: {}".format(len(source_list)))

    for i in range(len(source_list)):
        source_path = source_list[i]
        source_name = get_base_name(source_path)
        source_folder_name=source_name.split(".")[0]
        source_folder = os.path.join(save_path,source_folder_name)
        if not os.path.exists(source_folder):
            os.mkdir(source_folder)
        ##Copy the source image to the folder
        os.system("cp {} {}".format(source_path,source_folder))
        print("Copying {} to {}".format(source_path,source_folder))
        source_df = stacked_df[stacked_df['source']==source_path]
        ###Save the corresponding target images in the source folder
        # 
        for j in range(len(source_df)):
            target_path = source_df.iloc[j]['target']
            target_name = get_base_name(target_path)
            target_save_path = os.path.join(source_folder,target_name)
            target_type= "tk" if "tk" in target_path else "pr" 
            target_type = "partner" if ("partner" in target_path and target_type=="pr") else target_type
            os.system("cp {} {}".format(target_path,target_save_path))
            ##rename the target image to the source image _ j
            os.system("mv {} {}".format(target_save_path,os.path.join(source_folder,source_folder_name+"-var-"+target_type+str(j)+".png")))
            print("Copying {} to {}".format(target_path,target_save_path))        

    
    ###Make a df with counts of files in each folder
    folder_list = os.listdir(save_path)

    folder_count_df = pd.DataFrame(columns=['folder_name','count'])
    for folder in folder_list:
        folder_path = os.path.join(save_path,folder)
        count = len(os.listdir(folder_path))
        folder_count_df = folder_count_df.append({'folder_name':folder,'count':count},ignore_index=True)

    print(folder_count_df.head(100))


    ###Now split the dataset
    
    data_dir=save_path

    dataset_paths=glob(data_dir+"/*")

    split_prop = [0.6,0.2,0.2]

    train_paths = []
    test_paths = []
    val_paths = []

    ##SShuffle dataset paths
    random.shuffle(dataset_paths)


    for i in range(len(dataset_paths)):
        path = dataset_paths[i]
        if i < len(dataset_paths)*split_prop[0]:
            train_paths.append(path)
        elif i < len(dataset_paths)*(split_prop[0]+split_prop[1]):
            test_paths.append(path)
        else:
            val_paths.append(path)

    ##Save paths to json {images:[{file_name:file_i}] }

    train_list = []
    test_list = []
    val_list = []

    for i in range(len(train_paths)):
        path = train_paths[i]
        train_list.append({"file_name":path.split("/")[-1]})
    
    train_dict = {"images":train_list}

    for i in range(len(test_paths)):
        path = test_paths[i]
        test_list.append({"file_name":path.split("/")[-1]})
    
    test_dict = {"images":test_list}
    
    for i in range(len(val_paths)):
        path = val_paths[i]
        val_list.append({"file_name":path.split("/")[-1]})

    val_dict = {"images":val_list}


    
    ###SAve to json


    # save_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk_synth/splits/"
    save_path = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/"
    ###Make the folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    with open(os.path.join(save_path,"train.json"),"w") as f:
        json.dump(train_dict,f)

    with open(os.path.join(save_path,"test.json"),"w") as f:
        json.dump(test_dict,f)

    with open(os.path.join(save_path,"val.json"),"w") as f:
        json.dump(val_dict,f)

    print(train_dict.keys())

################Print paths
    train_list = []
    test_list = []
    val_list = []

    for i in range(len(train_paths)):
        path = train_paths[i]
        train_list.append({"file_name":path})
    
    train_dict = {"images":train_list}

    for i in range(len(test_paths)):
        path = test_paths[i]
        test_list.append({"file_name":path})
    
    test_dict = {"images":test_list}
    
    for i in range(len(val_paths)):
        path = val_paths[i]
        val_list.append({"file_name":path})

    val_dict = {"images":val_list}

    save_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/"

    ###Make the folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    with open(os.path.join(save_path,"train_paths.json"),"w") as f:
        json.dump(train_dict,f)

    with open(os.path.join(save_path,"test_paths.json"),"w") as f:
        json.dump(test_dict,f)

    with open(os.path.join(save_path,"val_paths.json"),"w") as f:
        json.dump(val_dict,f)

    print(train_dict.keys())



###ADd no match splits
###First add nomatch to the test and val paths
    no_match_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/nomatch/TK_nomatch_781.csv"
    no_match_df = pd.read_csv(no_match_data_path)
    no_match_df = no_match_df.dropna()
    no_match_df = no_match_df.reset_index(drop=True)

    ###Split the nomatch data into test and val
    split_prop = [0.5,0.5]

    no_match_test_df = no_match_df.iloc[:int(len(no_match_df)*split_prop[0])]
    no_match_val_df = no_match_df.iloc[int(len(no_match_df)*split_prop[0]):]

    no_match_val_paths=[]
    no_match_test_paths=[]
    for i in range(len(no_match_test_df)):
        row = no_match_test_df.iloc[i]
        no_match_test_paths.append(row["source"])
    
    for i in range(len(no_match_val_df)):
        row = no_match_val_df.iloc[i]
        no_match_val_paths.append(row["source"])
    
    test_paths_with_nomatch = test_paths + no_match_test_paths
    val_paths_with_nomatch = val_paths + no_match_val_paths

    ##Save no match val and test dicts
    test_list = []
    val_list = []

    for i in range(len(test_paths_with_nomatch)):
        path = test_paths_with_nomatch[i]
        test_list.append({"file_name":path})
    
    test_dict = {"images":test_list}

    for i in range(len(val_paths_with_nomatch)):
        path = val_paths_with_nomatch[i]
        val_list.append({"file_name":path})
    

    val_dict = {"images":val_list}

    ###SAve test and val dicts with nomatch
    with open(os.path.join(save_path,"test_paths_with_nomatch.json"),"w") as f:
        json.dump(test_dict,f)
    
    with open(os.path.join(save_path,"val_paths_with_nomatch.json"),"w") as f:
        json.dump(val_dict,f)

    print("Test and val dicts with nomatch saved")

    ####Now  save dict with only the base file name for each image
    test_list = []
    val_list = []
    
    for i in range(len(test_paths_with_nomatch)):
        path = test_paths_with_nomatch[i]
        test_list.append({"file_name":os.path.basename(path)})
    
    test_dict = {"images":test_list}

    for i in range(len(val_paths_with_nomatch)):
        path = val_paths_with_nomatch[i]
        val_list.append({"file_name":os.path.basename(path)})

    val_dict = {"images":val_list}

    ###SAve test and val dicts with nomatch
    with open(os.path.join(save_path,"test_with_nomatch_base.json"),"w") as f:
        json.dump(test_dict,f)
    
    with open(os.path.join(save_path,"val_with_nomatch_base.json"),"w") as f:
        json.dump(val_dict,f)




    
    ###Make a folder with only training images
    train_images_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_images/"
    
    ##Remove the folder if it exists
    if os.path.exists(train_images_folder):
        os.system("rm -r {}".format(train_images_folder))

    if not os.path.exists(train_images_folder):
        os.makedirs(train_images_folder)

    for i in range(len(train_paths)):
        path = train_paths[i]
        os.system("cp -r {} {}".format(path,train_images_folder))


    ###Make a folder with only one image from each folder in training images
    single_train_images_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/single_train_images/"
    
    if os.path.exists(single_train_images_folder):
        os.system("rm -r {}".format(single_train_images_folder))
    if not os.path.exists(single_train_images_folder):
        os.makedirs(single_train_images_folder)
    
    for i in range(len(train_paths)):
        path = train_paths[i]
        image_list = os.listdir(path)
        image_path = os.path.join(path,image_list[0])
        ##Make a folder with the same basename as the path
        folder_name = os.path.basename(path)
        folder_path = os.path.join(single_train_images_folder,folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        os.system("cp {} {}".format(image_path,folder_path))

    print("Done")
    

    
    


    
    




    ###Collect images from the source columns
