
###Preprocess data csv

import pandas as pd
import numpy as np
import json
import os
import sys

import faiss
import data_loaders
import train
from utils.datasets_utils import *
import itertools
from sklearn.model_selection import train_test_split
import random


def convert_to_text(unicode_string):
    return unicode_string.encode('ascii','ignore').decode('ascii')





def prepare_data(
    ground_truth_pr_tk="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/TK_matched_1207_appended.csv",
    ground_truth_pr_pr="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/PR_matched_1092_appended.csv",
    pr_partners_data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_partner.json",
    tk_data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_tk_title_dup_68352_clean_path.json",
    pr_title_data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_title_updated.json",
    output_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr.csv'
):
     ###Import the labelled data df
    ground_truth_pr_tk=pd.read_csv(ground_truth_pr_tk)

    ground_truth_pr_pr=pd.read_csv(ground_truth_pr_pr)
    
    # Load the ocr data
    with open(pr_partners_data) as f:
        pr_partners_data = json.load(f)

    with open(tk_data) as f:
        tk_data = json.load(f)

    with open(pr_title_data) as f:
        pr_title_data = json.load(f)


    ###Goal : Add ocr text to the ground truth df

    ###PR-TK
    print(len(ground_truth_pr_tk),"ground truth pr-tk pairs")

    ##Get OCR text for both source and target. source and target contain paths - simply get the text from path list for each data

    ground_truth_source_text=[v for k,v in pr_partners_data.items() ]
    ground_truth_target_text=[sublist[0] for sublist in tk_data ]
    print(ground_truth_target_text)
    ###Get the corresponding paths and make a df
    ground_truth_source_paths=[k for k,v in pr_partners_data.items()]
    ground_truth_target_paths=[sublist[1] for sublist in tk_data ]
    # print(ground_truth_source_paths)
    ##Build the dataframe source df
    source_df=pd.DataFrame(ground_truth_source_paths, columns=['source'])
    source_df['source_text_gt']=ground_truth_source_text
    ###Drop the duplicates
    source_df=source_df.drop_duplicates(subset=['source'])
    ##Build the dataframe target df
    target_df=pd.DataFrame(ground_truth_target_paths, columns=['target'])
    target_df['target_text_gt']=ground_truth_target_text
    target_df=target_df.drop_duplicates(subset=['target'])


    ####Merge with the ground truth df sequentially
    ground_truth_pr_tk=ground_truth_pr_tk.merge(source_df, on='source', how='left')
    print(len(ground_truth_pr_tk),"ground truth pr-tk pairs")

    ground_truth_pr_tk=ground_truth_pr_tk.merge(target_df, on='target', how='left')

        
    ###Now do it for PR-PR
    print(len(ground_truth_pr_pr),"ground truth pr-pr pairs")

    ##Get OCR text for both source and target. source and target contain paths - simply get the text from path list for each data

    ##Get the text for each source and target
    ground_truth_source_text=[v for k,v in pr_partners_data.items() ]
    ground_truth_target_text=[sublist[0] for sublist in pr_title_data  ]
    ###Get the corresponding paths and make a df
    ground_truth_source_paths=[k for k,v in pr_partners_data.items()]
    ground_truth_target_paths=[sublist[1] for sublist in pr_title_data  ]

    ##Build the dataframe source df
    source_df=pd.DataFrame(ground_truth_source_paths, columns=['source'])
    source_df['source_text_gt']=ground_truth_source_text
    ###Drop the duplicates
    source_df=source_df.drop_duplicates(subset=['source'])

    ##Build the dataframe target df
    target_df=pd.DataFrame(ground_truth_target_paths, columns=['target'])
    target_df['target_text_gt']=ground_truth_target_text
    target_df=target_df.drop_duplicates(subset=['target'])

    ##Merge with the ground truth df sequentially
    ground_truth_pr_pr=ground_truth_pr_pr.merge(source_df, on='source', how='left')
    print(len(ground_truth_pr_pr),"ground truth pr-pr pairs")

    ground_truth_pr_pr=ground_truth_pr_pr.merge(target_df, on='target', how='left')

    # ###Save both the dfs
    ground_truth_pr_tk.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/TK_matched_ocr_0303.csv", index=False,encoding='utf-8-sig')
    ground_truth_pr_pr.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_matched_ocr_0303.csv", index=False,encoding='utf-8-sig')


    print(len(ground_truth_pr_tk),"ground truth pr-tk pairs")
    print(len(ground_truth_pr_pr),"ground truth pr-pr pairs")

    ###For training the model, we want source to be linked to both pr and tk
    text_only_pr_tk=ground_truth_pr_tk[['source','target','source_text_gt','target_text_gt']]
    text_only_pr_pr=ground_truth_pr_pr[['source','target','target_text_gt']]

    ##Rename the columns
    text_only_pr_tk.columns=['source','target_tk','source_text_gt','target_text_gt_tk']
    text_only_pr_pr.columns=['source','target_pr','target_text_gt_pr']

    ##Merge the two dfs
    text_only_pr_tk_pr=text_only_pr_tk.merge(text_only_pr_pr, on='source', how='left')

    ##Save the df
    text_only_pr_tk_pr.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(len(text_only_pr_tk_pr),"ground truth pr-pr-tk pairs")

    return(text_only_pr_tk_pr)



    
def prep_labelled_data(
    data_path= "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr.csv",
    split_to_keep="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_paths.json",
    split_output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_train.csv",
):
    """Removes the test images from the data csv and saves the new csv"""

    
    data = pd.read_csv(data_path)

    ##Open train and val paths
    with open(split_to_keep) as f:
        split_linkage = json.load(f)



    ###Test paths
    split_linkage=split_linkage["images"]
    split_linkage=[subdict["file_name"].split("/")[-1] for subdict in split_linkage]
    print(split_linkage[0:5])



    ###Add a column to the dataframe with the base name only
    data['base_name']=data['source'].apply(lambda x: x.split("/")[-1].split(".")[0])
    print(data['base_name'][0:5])
    ###Subset the dataframe to only keep the paths in the split
    data=data[data['base_name'].isin(split_linkage)]
    print(len(data),"in split")
    ###Drop the base name column
    data=data.drop(columns=['base_name'])

    ##Drop duplicates
    data=data.drop_duplicates(subset=['source'])


    ###Reshape the dataset such that each row is a source-target pair. target_text_gt_tk, target_text_gt_pr, target_tk and target_pr get a different row with the same source.
    # ## two variables will be formed in the long form
    ###Essentially, the dataset needs to be converted to long form
    ###Python requires splitting and then metling
    ###Split the dataset into two
    data_long_target_paths=data[['source','source_text_gt','target_tk','target_pr']]
    data_long_target_text=data[['source','target_text_gt_tk','target_text_gt_pr']]
    ###Melt the two datasets
    data_long_target_paths=pd.melt(data_long_target_paths, id_vars=['source','source_text_gt'], var_name='target_type', value_name='target')
    ###Remove the target_type column
    data_long_target_paths=data_long_target_paths.drop(columns=['target_type'])
    data_long_target_text=pd.melt(data_long_target_text, id_vars=['source'], var_name='target_type', value_name='target_text_gt')
    ###Remove the target_type column
    data_long_target_text=data_long_target_text.drop(columns=['target_type'])
    ###Merge the two datasets
    data_long=data_long_target_paths.merge(data_long_target_text, on=['source'], how='left')
    ##REmove the rows where target_text_gt==""
    data_long=data_long[data_long['target_text_gt']!=""]
    ##Remove where there is an nan
    data_long=data_long.dropna()

    ###For each source, define a "class label" that is unique to the source 
    source_list=data_long['source'].unique()
    source_list=sorted(source_list)
    source_id=[i for i in range(len(source_list))]
    source_id_dict=dict(zip(source_list,source_id))
    data_long['source_id']=data_long['source'].apply(lambda x: source_id_dict[x])

    ###We now want to convert the df into a format that is suitable for training the model
    ###It has to have image_path,text,label where label=source_id
    ##For this, we need to split the target and sources while keeping the source_id
    ###Split the target
    data_long_target=data_long[['target','target_text_gt','source_id']]
    data_long_target.columns=['image_path','text','label']
    ###Split the source
    data_long_source=data_long[['source','source_text_gt','source_id']]
    data_long_source.columns=['image_path','text','label']

    ###Append the two dfs by concatenating them
    data_long=pd.concat([data_long_target,data_long_source], axis=0)

    ###Drop duplicates by all columns
    data_long=data_long.drop_duplicates()

    ###Save the new csv
    data_long.to_csv(split_output_path, index=False,encoding='utf-8-sig')
    print(len(data_long),"data_long")
    print("saved to",split_output_path)

    
    return data_long

def prep_labeled_test_data(
    data_path= "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr.csv",
    test_viz_linkage="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/test_paths_final.json",
    output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_test.csv"    
):
    """Removes the train-val images from the data csv and saves the new csv"""

    
    data = pd.read_csv(data_path)

    ##Test set path
    
    with open(test_viz_linkage) as f:
        test_viz_linkage = json.load(f)

    ###Test paths
    test_viz_linkage=test_viz_linkage["images"][0]

    ###Split by "/" and get the last element
    test_viz_linkage=[sublist.split("/")[-1] for sublist in test_viz_linkage]

    ###Add a column to the dataframe with the base name only
    data['base_name']=data['source'].apply(lambda x: x.split("/")[-1])

    ###Subset the dataframe to only contain only the test set
    data=data[data['base_name'].isin(test_viz_linkage)]

    ###Drop the base name column
    data=data.drop(columns=['base_name'])

    ##Drop duplicates
    data=data.drop_duplicates(subset=['source'])


    ###Reshape the dataset such that each row is a source-target pair. target_text_gt_tk, target_text_gt_pr, target_tk and target_pr get a different row with the same source.
    # ## two variables will be formed in the long form
    ###Essentially, the dataset needs to be converted to long form
    ###Python requires splitting and then metling
    ###Split the dataset into two
    data_long_target_paths=data[['source','source_text_gt','target_tk','target_pr']]
    data_long_target_text=data[['source','target_text_gt_tk','target_text_gt_pr']]
    ###Melt the two datasets
    data_long_target_paths=pd.melt(data_long_target_paths, id_vars=['source','source_text_gt'], var_name='target_type', value_name='target')
    ###Remove the target_type column
    data_long_target_paths=data_long_target_paths.drop(columns=['target_type'])
    data_long_target_text=pd.melt(data_long_target_text, id_vars=['source'], var_name='target_type', value_name='target_text_gt')
    ###Remove the target_type column
    data_long_target_text=data_long_target_text.drop(columns=['target_type'])
    ###Merge the two datasets
    data_long=data_long_target_paths.merge(data_long_target_text, on=['source'], how='left')
    ##REmove the rows where target_text_gt==""
    data_long=data_long[data_long['target_text_gt']!=""]
    ##Remove where there is an nan
    data_long=data_long.dropna()

    ###For each source, define a "class label" that is unique to the source 
    source_list=data_long['source'].unique()
    source_list=sorted(source_list)
    source_id=[i for i in range(len(source_list))]
    source_id_dict=dict(zip(source_list,source_id))
    data_long['source_id']=data_long['source'].apply(lambda x: source_id_dict[x])

    ###We now want to convert the df into a format that is suitable for training the model
    ###It has to have image_path,text,label where label=source_id
    ##For this, we need to split the target and sources while keeping the source_id
    ###Split the target
    data_long_target=data_long[['target','target_text_gt','source_id']]
    data_long_target.columns=['image_path','text','label']
    ###Split the source
    data_long_source=data_long[['source','source_text_gt','source_id']]
    data_long_source.columns=['image_path','text','label']

    ###Append the two dfs by concatenating them
    data_long=pd.concat([data_long_target,data_long_source], axis=0)

    ###Drop duplicates by all columns
    data_long=data_long.drop_duplicates()

    ###In the case of test data, we only need to keep source images (PR titles). They contain the string "dot_dect_1130/element_crop/" in the image_path
    data_long=data_long[data_long['image_path'].str.contains("pr_partner_crop")]
    
    
    ###Save the new csv
    data_long.to_csv(output_path, index=False,encoding='utf-8-sig')
    print(len(data_long),"data_long")
    print("saved to",output_path)

    
    return data_long

def make_data_with_hardnegs(
    train_df,
    output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_hardnegs.csv",
    k=8,
    batch_size=128,
    m=4,
    model=None,
    device="cpu"   
):
    """Make a csv with hardnegs. For each text-image pair, feed it into the model and get the embeddings. Then make an index and get k
    nearest neighbours. Then search these embeddings within the index to get k nearest neighbours for each of them. 
    Then make the dataframe with an anchor id."""

    ###Make a dataset and a data loader for the data

    ##First load the df if its a string
    if type(train_df)==str:
        train_df=pd.read_csv(train_df)
    
    ###Make a dataset
    train_dataset = data_loaders.TextImageDataset(train_df, transform=BASE_TRANSFORM)
    ###Make a dataloader
    train_loader = data_loaders.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)

    ###Get all embeddings
    ref_embeddings, ref_labels, ref_text, ref_image_paths = train.get_all_embeddings(train_loader, model, device)

    ###Make an index
    index = faiss.IndexFlatIP(ref_embeddings.shape[1])
    index.add(ref_embeddings)

    ###Get the k nearest neighbours
    D, I = index.search(ref_embeddings, k)

    ###Get the texts of the k nearest neighbours
    ref_text_knn=[ref_text[i] for i in I]

    ###Get the image paths of the k nearest neighbours
    ref_image_paths_knn=[ref_image_paths[i] for i in I]

    ###Get the labels of the k nearest neighbours
    ref_labels_knn=[ref_labels[i] for i in I]

    ###Anchor ids - for each I, make a list of m elements, where each element is the index of the anchor
    anchor_ids=[]
    for i in range(len(I)):
        anchor_ids.append([i]*m)
    ###Make a datarame and give an anchor id
    df=pd.DataFrame()
    df['anchor_id']=list(itertools.chain.from_iterable(anchor_ids))
    df['image_path']=list(itertools.chain.from_iterable(ref_image_paths_knn))
    df['text']=list(itertools.chain.from_iterable(ref_text_knn))
    df['label']=list(itertools.chain.from_iterable(ref_labels_knn))

    ###Rearrange the columns
    df=df[['image_path','text','label','anchor_id']]

    ###Save the df
    df.to_csv(output_path, index=False,encoding='utf-8-sig')

    return df


def prep_synthetic_data(input_path_list=["/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/ocr-as-retrieval/misc/multimodal_synth_0217/dataframes/Jake_noise/filled_noisy_2_ocr_ground_truth.csv",
                                         "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/ocr-as-retrieval/misc/multimodal_synth_0217/dataframes_0228/Jake_noise/filled_noisy_2_ocr_ground_truth.csv",
                                         "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/ocr-as-retrieval/misc/multimodal_synth_0217/dataframes/Ab_noise/filled_noisy_2_ocr_ground_truth.csv"],
                        output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_data.csv",
                        drop_prop=0.9):
    """Prepare the synthetic data"""
    ###Load the synthetic data from the list and concatenate them
    synthetic_data=pd.concat([pd.read_csv(i) for i in input_path_list], axis=0)
    ###Drop the rows with empty OCR_text
    synthetic_data=synthetic_data[synthetic_data['OCR_text'].notna()]

    ###remove the first column
    synthetic_data=synthetic_data.drop(columns=['Unnamed: 0'])
    ###Add the label. All rows with the same ground truth text will have the same label
    synthetic_data['label']=synthetic_data['ground_truth_text'].apply(lambda x: hash(x))

    ###Give a numeric id to each label
    label_list=synthetic_data['label'].unique()
    label_list=sorted(label_list)
    label_id=[i for i in range(len(label_list))]
    label_id_dict=dict(zip(label_list,label_id))
    synthetic_data['label']=synthetic_data['label'].apply(lambda x: label_id_dict[x])

    ###Rename Image to image_path
    synthetic_data=synthetic_data.rename(columns={'Image_path':'image_path'})

    ###Rename OCR_text to text
    synthetic_data=synthetic_data.rename(columns={'OCR_text':'text'})

    ###We only want to use the clean images
    ####Replace "/noise_Ab/" in image_path with "/clean/"
    synthetic_data['image_path']=synthetic_data['image_path'].apply(lambda x: x.replace("/noise_Ab/","/clean/images/"))

    ###Drop ground_truth_text
    synthetic_data=synthetic_data.drop(columns=['ground_truth_text'])

    print(len(synthetic_data),"rows in the synthetic data")

    ####The dataset is too large, drop some rows by labels. - sample 1-drop prop of the labels
    label_list=synthetic_data['label'].unique()
    label_list=sorted(label_list)

    ###Make a list of labels to drop
    ###Sample drop_prop of the labels
    labels_to_drop=random.sample(label_list, int(len(label_list)*drop_prop))

    ###Keep only the labels that are not in the list of labels to drop

    ###Drop the labels
    synthetic_data=synthetic_data[~synthetic_data['label'].isin(labels_to_drop)]

    print(len(synthetic_data),"rows in the synthetic data after dropping some labels")


    ###Save the df
    synthetic_data.to_csv(output_path, index=False,encoding='utf-8-sig')

    print(synthetic_data.head())

    ##Make train and test split - 80% train, 20% test by label
    train_data, test_data = train_test_split(synthetic_data, test_size=0.2, random_state=42, stratify=synthetic_data['label'])

    ###Save the train and test data
    train_data.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_train_data.csv", index=False,encoding='utf-8-sig')
    test_data.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_test_data.csv", index=False,encoding='utf-8-sig')

    print("Final length of the train data:",len(train_data))

    return synthetic_data,train_data,test_data


def make_synthetic_ref_data(output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_ref_data.csv"):
    """Make a reference dataset for the synthetic data"""
    ###Load the synthetic data
    synthetic_data=pd.read_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_data.csv")

    ###Make a reference dataset
    ref_data=synthetic_data.drop_duplicates(subset=['label'],keep='first')

    ###Save the ref data
    ref_data.to_csv(output_path, index=False,encoding='utf-8-sig')

    return ref_data






  





    





###run as script
if __name__ == "__main__":
    prepare_data()
    prep_labelled_data(
    data_path= "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr.csv",
    split_to_keep="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_paths.json",
    split_output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_train.csv",
)
    prep_labelled_data(
    data_path= "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr.csv",
    split_to_keep="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/val_paths.json",
    split_output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_val.csv",
)

    prep_labelled_data(
    data_path= "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr.csv",
    split_to_keep="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/test_paths.json",
    split_output_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_test.csv",
)


    # prep_synthetic_data(drop_prop=0)
    # make_synthetic_ref_data()


