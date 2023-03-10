####Run inference over all of tk


import torch
import torch.nn as nn
from glob import glob

import data_loaders
import encoders

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, StepLR


import pandas as pd
import numpy as np
import wandb
from utils.datasets_utils import *
from tqdm.autonotebook import trange
import sentence_transformers

from  sentence_transformers import SentenceTransformer 
import faiss 
from tqdm import tqdm

import json
import matplotlib.pyplot as plt
from PIL import Image

def tokenize_sentences(sbert_model,sentences):
    tokenised_sentences = sbert_model.tokenize(sentences)
    return tokenised_sentences


def get_all_embeddings(dataset, model, batch_size=128,sbert_model=None,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    for batch_idx, (text, image_data, labels, anchor_ids) in enumerate(tqdm(dataset)):
        ####Unsquueze the image data
        image_data = image_data.to(device)
        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = tokenize_sentences(sbert_model,text)
         ##Send to device
        text_features = sentence_transformers.util.batch_to_device(text_features,torch.device('cuda'))
        with torch.no_grad():
            embeddings = model(image_data, text_features)
        if batch_idx == 0:
            all_embeddings = embeddings
            all_labels = labels
            all_text=text
        else:
            all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_text=all_text+text
    return all_embeddings, all_labels, all_text


###Run as script
if __name__ == "__main__":



    def main(path_to_weights="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/enc_best_e_both_unfrozen_mlp_2_final_4.pth"):
        ##Setup
        ##Load test data 
        test_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_test.csv"
        test_data = pd.read_csv(test_data_path)
        ###Drop duplicates by image_path and text
        test_data=test_data.drop_duplicates(subset=['image_path','text'])
        ##Add a column for the label as the 3rd column. All values = 1
        # test_data.insert(2, "label", 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##Load the model
        ###image encoder
        auto_timm_model="vit_base_patch16_224.dino"
        image_model_cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/vit_word_nohn_japan_center_places_20000_finetuned/enc_best_e.pth"
        image_encoder = encoders.AutoVisionEncoderFactory("timm", auto_timm_model)
        image_enc=image_encoder.load(image_model_cp_path)
        image_enc=image_enc.to(device)
        ###text encoder
        trained_sbert_model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/language_model_linkage/output/2023-02-14_20-02-05/800"
        text_encoder = encoders.SBERTEncoder(trained_sbert_model_path)
        sbert_model_for_tokenizer=SentenceTransformer(trained_sbert_model_path)
        text_encoder=text_encoder.to(device)

        ###Create the multimodal model
        model = encoders.MultiModalEncoder(image_enc, text_encoder,num_layers=1)

        if path_to_weights!="":
            model.load_state_dict(torch.load(path_to_weights))
        model=model.to(device)

        model.eval()
        ####Now prep the tk universe. IT also needs to follow the format image_path,text
        tk_data='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/homo_match_dataset/japan/effocr_tk_title.json'

        with open(tk_data) as f:
            tk_data = json.load(f)
        
        
        ground_truth_target_text=[sublist[0] for sublist in tk_data if sublist[1] ]
        ground_truth_target_images=[sublist[1] for sublist in tk_data if sublist[1] ]

        tk_data_df=pd.DataFrame({'image_path':ground_truth_target_images,'text':ground_truth_target_text})

        tk_data_df=tk_data_df.drop_duplicates(subset=['image_path','text'])
        tk_data_df=tk_data_df.dropna()
        tk_data_df=tk_data_df.reset_index(drop=True)

        ##Add a label column with all as 1
        tk_data_df.insert(2, "label", 1)

        # tk_data_df=tk_data_df.sample(20000)

        ###Now we make a dataset for tk
        tk_dataset=data_loaders.TextImageDataset(tk_data_df,img_transform=BASE_TRANSFORM)

        ###Now make a dataloader for tk
        tk_dataloader = torch.utils.data.DataLoader(tk_dataset, batch_size=400, shuffle=False, num_workers=16)

        ###embed tk using the model
        tk_embeddings,_,_=get_all_embeddings(tk_dataloader,model,batch_size=400,sbert_model=sbert_model_for_tokenizer)

        ####Similarly, embed the test data
        test_dataset=data_loaders.TextImageDataset(test_data,img_transform=BASE_TRANSFORM)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

        test_embeddings,_,_=get_all_embeddings(test_dataloader,model,batch_size=128,sbert_model=sbert_model_for_tokenizer)


        ####Now we need to build an index for tk
        index = faiss.IndexFlatIP(tk_embeddings.shape[1])
        index.add(tk_embeddings.cpu().numpy())

        ####Now we need to query the index
        k=1
        D, I = index.search(test_embeddings.cpu().numpy(), k) # actual search
        
        ####Now we need to compare it with ground truth
        test_data['tk_image_path']=tk_data_df.iloc[I[:,0]]['image_path'].values

        ####Drop the text column from test data
        test_data=test_data.drop(columns=['text'])

        ##Rename the image_path column to target and tk_image_path to source
        test_data=test_data.rename(columns={'image_path':'source','tk_image_path':'target_matched'})

        ###Drop duplicates by target and source
        test_data=test_data.drop_duplicates(subset=['target_matched','source'])

        ###Load the ground truth labels
        ground_truth_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk/TK_matched.csv"

        ground_truth=pd.read_csv(ground_truth_path)

        ###Now merge with ground truth to find accuracy by source
        merged_df=pd.merge(test_data,ground_truth,on=['source'],how='left')

        ###If the target_matched is equal to target, then it is a match
        merged_df['match']=merged_df['target_matched']==merged_df['target']

        merged_df=merged_df.drop_duplicates(subset=['source'])

        ###Now we can find the total number of matches
        total_matches=merged_df['match'].sum()

        ###Now we can find the total number of records
        total_records=merged_df.shape[0]

        ###Now we can find the accuracy
        accuracy=total_matches/total_records

        print(accuracy)

        ###Drop duplicates by source

        ##GEt incorrect matches
        incorrect_matches=merged_df[merged_df['match']==0]

        ###Visualise the incorrect matches
        ###Get basepaths
        source_root_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/dot_dect_1130/element_crop/'
        target_root_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/tk_title_img_1025_v2/'   
        ###Reappend the basepath to the source and target by appending source_path and target_path
        # incorrect_matches['source']=incorrect_matches['source'].apply(lambda x: os.path.join(source_root_path, x))
        # incorrect_matches['target']=incorrect_matches['target'].apply(lambda x: os.path.join(target_root_path, x))

        ###Reappend the basepath to the source and target by appending source_path and target_path
        incorrect_matches['source']=incorrect_matches['source']
        incorrect_matches['target']=incorrect_matches['target']

        ###chunk the incorrect matches into parts consisting of 5 images
        incorrect_matches_chunks=[incorrect_matches[i:i+5] for i in range(0,incorrect_matches.shape[0],5)]






        ##Visualize the incorrect matches using plt and Image and the associated ground truth. All should be in the same image
        def visualize_incorrect_matches(incorrect_matches_df,c_num=0):
            c_num=str(c_num)
        ##Visualize the incorrect matches using plt and Image
            fig, ax = plt.subplots(len(incorrect_matches_df), 3, figsize=(20, 20))
            print(len(incorrect_matches_df["source"].unique()))
            for i,img_name in enumerate(incorrect_matches_df["source"].unique()):
                img = Image.open(img_name)
                ax[i,0].imshow(img)
                ax[i,0].axis('off')
                # ax[i,0].set_title("Source")
                img = Image.open(incorrect_matches_df[incorrect_matches_df["source"]==img_name]["target"].iloc[0])
                ax[i,1].imshow(img)
                ax[i,1].axis('off')
                # ax[i,1].set_title("Target")
                img = Image.open(incorrect_matches_df[incorrect_matches_df["source"]==img_name]["target_matched"].iloc[0])
                ax[i,2].imshow(img)
                ax[i,2].axis('off')

            image_path=f'./incorrect_matches_{c_num}.png'
            ##Save the incorrect matches
            plt.savefig(image_path,dpi=600)

        ###Visualize the incorrect matches
        # for i,incorrect_matches_chunk in enumerate(incorrect_matches_chunks):
        #     visualize_incorrect_matches(incorrect_matches_chunk,c_num=i)

        print(accuracy)


    ####Get all model paths
    model_paths=glob.glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/*")

    ###Should contain .pth
    model_paths=[model_path for model_path in model_paths if '.pth' in model_path]

    ###Add an empty string to the first position
    model_paths.insert(0,'')

    ###Run the main function for each model. If error occurs, then skip
    # for model_path in model_paths:
    #     try:
    #         main(model_path)
    #     except:
    #         print(f"Error occured for model path {model_path}")
    #         pass
    main("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/enc_best_e_both_frozen_mlp_2_final_4.pth")
