####CLIPPERR Inference script

###Japanese CLIP model -trials

import io
import requests
from PIL import Image
import torch
import japanese_clip as ja_clip
import pandas as pd
import numpy as npf
from utils.datasets_utils import *
import data_loaders 
from tqdm import tqdm
import json
import faiss
from timm.models import load_state_dict
import argparse

import matplotlib.pyplot as plt
from datetime import datetime



def convert_to_text(unicode_string):
    return unicode_string.encode('ascii','ignore').decode('ascii')

def get_pr_partner_universe(pr_partner_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_partner.json"):
        
        with open(pr_partner_path) as f:
            pr_partner_data = json.load(f)
        
        
        text_list= list(pr_partner_data.values())
        image_list= list(pr_partner_data.keys())
    
        pr_partner_data_df=pd.DataFrame({'image_path':image_list,'text':text_list})
    
        pr_partner_data_df=pr_partner_data_df.drop_duplicates(subset=['image_path','text'])
        pr_partner_data_df=pr_partner_data_df.dropna()
        pr_partner_data_df=pr_partner_data_df.reset_index(drop=True)
    
        return pr_partner_data_df

def get_tk_universe(tk_data_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_tk_title_dup_68352_clean_path.json'):
    
    with open(tk_data_path) as f:
        tk_data = json.load(f)
    
    
    ground_truth_target_text=[sublist[0] for sublist in tk_data if sublist[1] ]
    ground_truth_target_images=[sublist[1] for sublist in tk_data if sublist[1] ]

    tk_data_df=pd.DataFrame({'image_path':ground_truth_target_images,'text':ground_truth_target_text})

    tk_data_df=tk_data_df.drop_duplicates(subset=['image_path','text'])
    tk_data_df=tk_data_df.dropna()
    tk_data_df=tk_data_df.reset_index(drop=True)
    # tk_data_df=tk_data_df.sample(300)
    return tk_data_df

def get_image_text_embeddings(data_loader,model,device,pooling_type='mean',im_wt=0.5):
    # model.to(device)
    # model.train()
    for batch_idx, (text, image_data, labels, image_path) in tqdm(enumerate(data_loader)):

            labels = labels.to(device)

            ####Unsquueze the image data
            image_data = image_data.to(device)

            ### text is a tuple of strings, we need to convert it to a tensor
            text=list(text)
            text_features = ja_clip.tokenize(text,tokenizer=tokenizer)

            for key in text_features.keys():
                text_features[key]=text_features[key].to(device)
            

            with torch.no_grad():
    

                model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                        attention_mask=text_features["attention_mask"], position_ids=text_features["position_ids"])
                image_embeds, text_embeds = model_output["image_embeds"], model_output["text_embeds"]

                # final_embeds=torch.cat((image_embeds,text_embeds),dim=1)
                ###MEan of the two embedding
                if pooling_type=='mean':
                    final_embeds=(image_embeds*im_wt +  text_embeds*(1-im_wt))
                elif pooling_type=='text':
                    final_embeds=text_embeds
                elif pooling_type=='image':
                    final_embeds=image_embeds
                # final_embeds=text_embeds
                ###Max of the two embeddings
                # final_embeds=torch.max(image_embeds,text_embeds)
                final_embeds=final_embeds/torch.norm(final_embeds, dim=1, keepdim=True)

                ####
                if batch_idx == 0:
                    all_embeddings = final_embeds
                    all_labels = labels
                    all_text=text
                    all_paths=image_path
                else:
                    all_embeddings = torch.cat((all_embeddings, final_embeds), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)
                    all_text=all_text+text
                    all_paths=all_paths+image_path

    return all_embeddings, all_labels, all_text, all_paths



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_similarity_torch(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def clean_checkpoint(checkpoint, use_ema=True, clean_aux_bn=False,clean_net=True):
    # Load an existing checkpoint to CPU, strip everything but the state_dict and re-save
    if checkpoint and os.path.isfile(checkpoint):
        print("=> Loading checkpoint '{}'".format(checkpoint))
        state_dict = load_state_dict(checkpoint, use_ema=use_ema)
        new_state_dict = {}
        for k, v in state_dict.items():
            if clean_aux_bn and 'aux_bn' in k:
                # If all aux_bn keys are removed, the SplitBN layers will end up as normal and
                # load with the unmodified model using BatchNorm2d.
                continue
            # name = k[7:] if k.startswith('module.') else k
            # new_state_dict[name] = v
            
            name = k[7:] if (k.startswith('module.') and clean_net==True) else k
            new_state_dict[name] = v
        print("=> Loaded state_dict from '{}'".format(checkpoint))
        return new_state_dict
    else:
        print("Error: Checkpoint ({}) doesn't exist".format(checkpoint))
        return ''

def get_pr_title_universe(pr_title_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_title_updated.json"):
    
    with open(pr_title_path) as f:
        pr_title_data = json.load(f)
    
    
    text_list=[convert_to_text(sublist[0]) for sublist in pr_title_data]
    image_list=[sublist[1] for sublist in pr_title_data]

    pr_title_data_df=pd.DataFrame({'image_path':image_list,'text':text_list})

    pr_title_data_df=pr_title_data_df.drop_duplicates(subset=['image_path','text'])
    pr_title_data_df=pr_title_data_df.dropna()
    pr_title_data_df=pr_title_data_df.reset_index(drop=True)



    return pr_title_data_df

    
def prep_test_universe_data(ocr_tk_match_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/TK_matched_ocr_0303.csv",
                            test_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_test.csv"):
        
    ocr_tk_match_df=pd.read_csv(ocr_tk_match_path)

    ##GEt only targets realted variables
    ocr_tk_match_df=ocr_tk_match_df[['target','target_text_gt']]

    ##Rename columns
    ocr_tk_match_df=ocr_tk_match_df.rename(columns={'target':'image_path','target_text_gt':'text'})

    ##Load test data 
    tk_universe=get_tk_universe()
    
    test_data = pd.read_csv(test_data_path)
    ###Drop duplicates by image_path and text
    test_data=test_data.drop_duplicates(subset=['image_path','text'])

    ##Keep only partner related image paths
    test_data=test_data[test_data['image_path'].str.contains("partner")]
    
    ###Drop duplicates by image_path and text
    tk_universe=tk_universe.drop_duplicates(subset=['image_path','text'])

    ##Add a label column
    ##Add a label column with all as 1
    tk_universe.insert(2, "label", 1)

    test_data=test_data.reset_index(drop=True)
    tk_universe=tk_universe.reset_index(drop=True)

    print("test data size: {}".format(len(test_data)))
    print("tk universe size: {}".format(len(tk_universe)))

    test_data=data_loaders.TextImageDataset(test_data, img_transform=CLIP_BASE_TRANSFORM)
    tk_universe=data_loaders.TextImageDataset(tk_universe,img_transform=CLIP_BASE_TRANSFORM)

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False, num_workers=16)
    tk_universe_loader = torch.utils.data.DataLoader(tk_universe, batch_size=500, shuffle=False, num_workers=16)

    return test_data_loader, tk_universe_loader

def prep_partner_tk_loaders(pr_partner_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_partner.json",
                            tk_data_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_tk_title_dup_68352_clean_path.json'):
    pr_partner_universe=get_pr_partner_universe(pr_partner_path)
    tk_universe=get_tk_universe(tk_data_path)

    pr_partner_universe.insert(2, "label", 1)
    tk_universe.insert(2, "label", 1)

    pr_partner_universe=pr_partner_universe.reset_index(drop=True)
    tk_universe=tk_universe.reset_index(drop=True)

    pr_partner_universe=data_loaders.TextImageDataset(pr_partner_universe, img_transform=CLIP_BASE_TRANSFORM)
    tk_universe=data_loaders.TextImageDataset(tk_universe,img_transform=CLIP_BASE_TRANSFORM)

    pr_partner_universe_loader = torch.utils.data.DataLoader(pr_partner_universe, batch_size=500, shuffle=False, num_workers=16)
    tk_universe_loader = torch.utils.data.DataLoader(tk_universe, batch_size=500, shuffle=False, num_workers=16)

    return pr_partner_universe_loader, tk_universe_loader



def get_clipping_embeddings(test_data_loader,tk_universe_loader,model,device,pooling_type="mean",im_wt=0.5):
    with torch.no_grad():

        test_embeddings, test_labels, test_text, test_paths = get_image_text_embeddings(test_data_loader,model, device,pooling_type=pooling_type,im_wt=0.5)
        ###Print test labels and text
        print("Test", test_labels[:10], test_text[:10])
        tk_universe_embeddings, tk_universe_labels, tk_universe_text, tk_paths = get_image_text_embeddings(tk_universe_loader,model, device,pooling_type=pooling_type,im_wt=0.5)
        print("test embeddings shape: {}".format(test_embeddings.shape))

        output_dict={"test_embeddings":test_embeddings,
                     "test_labels":test_labels,
                     "test_text":test_text,
                     "test_paths":test_paths,
                     "tk_universe_embeddings":tk_universe_embeddings,
                     "tk_universe_labels":tk_universe_labels,
                     "tk_universe_text":tk_universe_text,
                     "tk_paths":tk_paths
                     }

        return output_dict

def get_matches_using_faiss(clipping_rl_output_dict,output_df_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/test_tk_match.csv",gpu_faiss=False):
    "FAISS matching"
    
    ##Calculate time using datetime
    start_time=datetime.now()
    print("Start time: {}".format(start_time))
    ###Make an index using tk universe
    ##Parse output dict
    test_embeddings=clipping_rl_output_dict["test_embeddings"]
    test_text=clipping_rl_output_dict["test_text"]
    test_paths=clipping_rl_output_dict["test_paths"]
    tk_universe_embeddings=clipping_rl_output_dict["tk_universe_embeddings"]
    tk_universe_text=clipping_rl_output_dict["tk_universe_text"]
    tk_paths=clipping_rl_output_dict["tk_paths"]

    if gpu_faiss:
        ###USe gpu faiss
        res= faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, test_embeddings.shape[1])
        index.add(tk_universe_embeddings.cpu())
        D, I = index.search(test_embeddings.cpu(), 1)
        ##Convert D,I to cpu
        # D=D.cpu().numpy()
        # I=I.cpu().numpy()

    

    else:
        index = faiss.IndexFlatIP(test_embeddings.shape[1])
        index.add(tk_universe_embeddings.cpu().numpy())
    ###Get top 1 match
        D, I = index.search(test_embeddings.cpu().numpy(), 1)

    ###Print corresponding text of test and tk universe match
    end_time=datetime.now()
    time_taken=end_time-start_time
    print("Time taken: {}".format(time_taken))
    ###Make a dataframe of test image path and tk universe match path
    test_tk_match=pd.DataFrame()
    test_tk_match["source"]=test_paths
    test_tk_match["matched_tk_path"]=[tk_paths[i[0]] for i in I]
    test_tk_match["test_text"]=test_text
    test_tk_match["matched_tk_text"]=[tk_universe_text[i[0]] for i in I]
    ###GEt distances as well
    test_tk_match["distance"]=[i[0] for i in D]

    
    ##Save the df here
    test_tk_match.to_csv(output_df_path,index=False,encoding="utf-8-sig")

    return test_tk_match 


def calculate_match_accuracy(test_match_df,
                             ground_truth_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/TK_matched_1207_appended.csv",
                             incorrect_match_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/incorrect_matches.csv"):

    ###Now merge with test data
    ground_truth=pd.read_csv(ground_truth_path)

    ###Now merge with ground truth on source
    test_tk_match=pd.merge(test_match_df,ground_truth,on="source",how="left")



    ##Now calculate accuracy if matched_tk_path is same as target
    test_tk_match["accuracy"]=test_tk_match.apply(lambda x: 1 if x["matched_tk_path"]==x["target"] else 0,axis=1)

    print(test_tk_match.head(5))

    print("Accuracy: {}".format(test_tk_match["accuracy"].sum()/len(test_tk_match)))

    incorrect_match_df=test_tk_match[test_tk_match["accuracy"]==0]

    ###
    # incorrect_matches_chunks=[incorrect_match_df[i:i+5] for i in range(0,incorrect_match_df.shape[0],5)]
    ##Reset index
    incorrect_match_df=incorrect_match_df.reset_index(drop=True)



    incorrect_matches_chunks=[incorrect_match_df.filter(items=range(i,i+5),axis=0) for i in range(0,incorrect_match_df.shape[0],5) ]
    
    ##IF a chunk is of length 1, then append a row of all NaNs = columns of incorrect_match_df
    for i in range(len(incorrect_matches_chunks)):
        if incorrect_matches_chunks[i].shape[0]==1:
            incorrect_matches_chunks[i]=incorrect_matches_chunks[i].append(pd.DataFrame(np.nan, index=[0], columns=incorrect_match_df.columns))

        
  
    ##Save incorrect match_Df
    ##REname some columns and drop some
    ###Keep only source,matched_tk_path,target,label
    incorrect_match_df=incorrect_match_df[["source","matched_tk_path","target"]]
    incorrect_match_df.columns=["source","target_matched","target"]
    incorrect_match_df.to_csv(incorrect_match_path,index=False,encoding="utf-8-sig")


    return incorrect_matches_chunks



def get_accuracy_with_no_match(test_match_df,
                             ground_truth_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/TK_matched_1207_appended.csv",
                             threshold=0.5):
    ###Now merge with test data
    ground_truth=pd.read_csv(ground_truth_path)

    ###Now merge with ground truth on source
    test_tk_match=pd.merge(test_match_df,ground_truth,on="source",how="left")

    ##Now calculate accuracy if matched_tk_path is same as target if distance (similarity) is less than threshold
    test_tk_match["accuracy"]=test_tk_match.apply(lambda x: 1 if x["matched_tk_path"]==x["target"] and x["distance"]>=threshold else 0,axis=1)

    ###No match accuracy calculation requires an adjusted denominator
    adjusted_denominator=len(test_tk_match[test_tk_match["distance"]>=threshold])

    ####Calculate accuracy!
    print("Accuracy: {}".format(test_tk_match["accuracy"].sum()/adjusted_denominator))

    return test_tk_match



    




    
def visualize_incorrect_matches(incorrect_matches_df_chunk,c_num=0,img_prefix=""):
    c_num=str(c_num)
##Visualize the incorrect matches using plt and Image. Also print the distance
    fig, ax = plt.subplots(len(incorrect_matches_df_chunk), 3, figsize=(20, 20))
    print(len(incorrect_matches_df_chunk["source"].unique()))
    for i,img_name in enumerate(incorrect_matches_df_chunk["source"].unique()):
        ##If the image name is nan, then skip
        if type(img_name)!=str:
            continue
        img = Image.open(img_name)
        ax[i,0].imshow(img)
        ax[i,0].axis('off')
        # ax[i,0].set_title("Source")
        img = Image.open(incorrect_matches_df_chunk[incorrect_matches_df_chunk["source"]==img_name]["target"].iloc[0])
        ax[i,1].imshow(img)
        ax[i,1].axis('off')
        # ax[i,1].set_title("Target")
        img = Image.open(incorrect_matches_df_chunk[incorrect_matches_df_chunk["source"]==img_name]["matched_tk_path"].iloc[0])
        ax[i,2].imshow(img)
        ax[i,2].axis('off')
        ###Also print the distance
        ax[i,2].set_title("Distance: {}".format(incorrect_matches_df_chunk[incorrect_matches_df_chunk["source"]==img_name]["distance"].iloc[0]))


    image_path=f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/{img_prefix}_incorrect_matches_{c_num}.png'
    ##Save the incorrect matches
    plt.savefig(image_path,dpi=600)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--pooling_type", type=str, default="mean")
    parser.add_argument("--img_wt", type=float, default=0.5)
    parser.add_argument("--output_prefix", type=str, default="text_only")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--infer_partnertk", action="store_true")
    parser.add_argument("--ocr_result", type=str, default="effocr")
    parser.add_argument("--use_gpu_faiss", action="store_true")

    args = parser.parse_args()

    if args.split=="test":
        split_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_test.csv"
    elif args.split=="val":
        split_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/PR_TK_matched_ocr_only_val.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)

    if args.checkpoint_path is not None:
        model.load_state_dict(clean_checkpoint(args.checkpoint_path))
    else : 
        print("Using base japanese CLIP model rinna/japanese-clip-vit-b-16")
        

    model.to(device)
    tokenizer = ja_clip.load_tokenizer()

    if args.infer_partnertk:

        if args.ocr_result=="effocr":
            pr_partner_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_pr_partner.json"
            tk_data_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/effocr_tk_title_dup_68352_clean_path.json'
        elif args.ocr_result=="gcv":
            pr_partner_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/gcv_pr_partner.json"
            tk_data_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/ocr_json/gcv_tk_title_dup_68352_clean_path.json'
           

        partner_loader,tk_universe_loader=  prep_partner_tk_loaders(pr_partner_path,tk_data_path)
        clippings_linkage_output=get_clipping_embeddings(partner_loader,tk_universe_loader,model,device,pooling_type=args.pooling_type,im_wt=args.img_wt)
        ###Make the match df
        if args.use_gpu_faiss:

            test_match_df=get_matches_using_faiss(clippings_linkage_output,output_df_path=f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/{args.output_prefix}_partner_tk_match.csv",gpu_faiss=True)
        else:
            test_match_df=get_matches_using_faiss(clippings_linkage_output,output_df_path=f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/{args.output_prefix}_partner_tk_match.csv",gpu_faiss=False)
        
        ##BReak the script here
        print("Partner TK matches saved to disk. Exiting script")
        exit()



    ##Get test and universe loaders
    test_data_loader,tk_universe_loader=prep_test_universe_data(ocr_tk_match_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/TK_matched_ocr_0303.csv",
                            test_data_path=split_path)

    ##Get clippings embeddings - clippings output dict
    clippings_linkage_output=get_clipping_embeddings(test_data_loader,tk_universe_loader,model,device,pooling_type=args.pooling_type,im_wt=args.img_wt)

    ###Make the match df
    test_match_df=get_matches_using_faiss(clippings_linkage_output,output_df_path=f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/{args.output_prefix}_test_tk_match.csv")
    
    ###Get the incorrect matches and calulate accuracy
    incorrect_matches_chunks=calculate_match_accuracy(test_match_df,
                             ground_truth_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/labelled_data/matched/TK_matched_1207_appended.csv",
                             incorrect_match_path=f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/mm_dir/{args.output_prefix}_incorrect_matches.csv")
    
    ###Visualize incorrect matches
    for i,incorrect_matches_chunk in enumerate(incorrect_matches_chunks):
        visualize_incorrect_matches(incorrect_matches_chunk,c_num=i,img_prefix=args.output_prefix)


        



        


        



