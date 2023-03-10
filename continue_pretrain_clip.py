####Continue pretrain clip
###This script is to continue pretraining clip on the japanese dataset. Our dataset is a df of image-text pairs

import pandas as pd
import numpy as np
import wandb
from utils.datasets_utils import *
from tqdm.autonotebook import trange
import sentence_transformers

from  sentence_transformers import SentenceTransformer 
import faiss 
from tqdm import tqdm

import math
import json
import argparse

from sklearn.model_selection import train_test_split


import io
import requests
from PIL import Image
import torch
import japanese_clip as ja_clip


from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, StepLR

from torch import nn
 
import wandb

from utils.datasets_utils import *


import data_loaders
from transformers import AutoModel
import encoders




def convert_to_text(unicode_string):
    return unicode_string.encode('ascii','ignore').decode('ascii')


def get_tk_universe(tk_data_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/homo_match_dataset/japan/effocr_tk_title.json'):
    
    with open(tk_data_path) as f:
        tk_data = json.load(f)
    
    
    ground_truth_target_text=[sublist[0] for sublist in tk_data if sublist[1] ]
    ground_truth_target_images=[sublist[1] for sublist in tk_data if sublist[1] ]

    tk_data_df=pd.DataFrame({'image_path':ground_truth_target_images,'text':ground_truth_target_text})

    tk_data_df=tk_data_df.drop_duplicates(subset=['image_path','text'])
    tk_data_df=tk_data_df.dropna()
    tk_data_df=tk_data_df.reset_index(drop=True)

    return tk_data_df

def get_pr_title_universe(pr_title_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/homo_match_dataset/japan/effocr_pr_title.json"):
    
    with open(pr_title_path) as f:
        pr_title_data = json.load(f)
    
    
    text_list=[convert_to_text(sublist[0]) for sublist in pr_title_data]
    image_list=[sublist[1] for sublist in pr_title_data]

    pr_title_data_df=pd.DataFrame({'image_path':image_list,'text':text_list})

    tk_data_df=tk_data_df.drop_duplicates(subset=['image_path','text'])
    tk_data_df=tk_data_df.dropna()
    tk_data_df=tk_data_df.reset_index(drop=True)



    return pr_title_data_df

def get_pr_partner_universe(pr_partner_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/homo_match_dataset/japan/partners.json"):
        
        with open(pr_partner_path) as f:
            pr_partner_data = json.load(f)
        
        
        text_list= [subdict['partner_text'] for subdict in pr_partner_data]
        image_list=[subdict['partner_path'] for subdict in pr_partner_data]
    
        pr_partner_data_df=pd.DataFrame({'image_path':image_list,'text':text_list})
    
        tk_data_df=tk_data_df.drop_duplicates(subset=['image_path','text'])
        tk_data_df=tk_data_df.dropna()
        tk_data_df=tk_data_df.reset_index(drop=True)
    
        return pr_partner_data_df

def get_pr_partner_test_data(test_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_test.csv"):
        test_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_test.csv"
        test_data = pd.read_csv(test_data_path)
        ###Drop duplicates by image_path and text
        test_data=test_data.drop_duplicates(subset=['image_path','text'])
        test_data=test_data.reset_index(drop=True)

        return test_data


def eval_clip(val_loader,model,tokenizer):
    print("Evaluating the model - clip loss")
    model.eval()
    loss_list=[]
    with torch.no_grad():
        for batch_idx, (text, image_data, labels, image_path) in enumerate(val_loader):
            labels = labels.to(device)
            labels= torch.arange((labels.shape[0])).to(device)
            ###Unsquueze the image data
            image_data = image_data.to(device)

            ### text is a tuple of strings, we need to convert it to a tensor
            text=list(text)
            text_features = ja_clip.tokenize(text,tokenizer=tokenizer)

            for key in text_features.keys():
                text_features[key]=text_features[key].to(device)

            model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                attention_mask=text_features["attention_mask"], position_ids=text_features["position_ids"])
            logits_per_image, logits_per_text = model_output["logits_per_image"], model_output["logits_per_text"]

            loss = (img_loss(logits_per_image, labels) + text_loss(logits_per_text, labels))/2
            loss_list.append(loss.item())

    mean_loss= np.mean(loss_list)
    wandb.log({"val_loss":mean_loss})
    return mean_loss            





def prep_labelled_data(data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_train.csv"):
     ####Load the image+text paired data (CSV)
        
        data=pd.read_csv(data_path)
        print("original size of data: {}".format(len(data)))

        ###drop duplicates if image path and text are the same
        data=data.drop_duplicates(subset=['image_path','text'])

        ###Drop duplicates by "pr" and "text". Image is "pr" if image_path contains dot_dect_1130/element_crop/

        ##Create a pr variable 1 if image_path contains dot_dect_1130/element_crop/
        data['pr']=data['image_path'].apply(lambda x: 1 if "dot_dect_1130/element_crop/" in x else 0)

        ##Drop duplicate rows when pr=1 and text is the same. 
        split_data=data[data['pr']==1].drop_duplicates(subset=['text'])

        ##Merge the split data with the data where pr=0
        data=pd.concat([split_data,data[data['pr']==0]])

        print("post processing size of data: {}".format(len(data)))
        ##Drop pr
        data=data.drop(columns=['pr'])


        ###Split the data into train and val just using train test split on label
        all_unique_labels=data['label'].unique()

        ###Take 80% of the labels for train and 20% for val
        np.random.seed(seed=42)
        train_labels=np.random.choice(all_unique_labels,int(len(all_unique_labels)*0.8),replace=False)
        val_labels=[x for x in all_unique_labels if x not in train_labels]

        ###Get the train and val data
        train_data=data[data['label'].isin(train_labels)]
        val_data=data[data['label'].isin(val_labels)]

        return train_data,val_data,data


def prep_synth_data(data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_train_data.csv"):
    
        ####Load the image+text paired data (CSV)
        data=pd.read_csv(data_path)
        print("original size of data: {}".format(len(data)))

        ###drop duplicates if image path and text are the same
        data=data.drop_duplicates(subset=['image_path','text'])

        ###Drop if na
        data=data.dropna()
        print("post processing size of data: {}".format(len(data)))

        ###Conver labels to negative - subtract from 0. This will help distinguish synthetic data from real data
        data['label']=data['label'].apply(lambda x: 0-x)

        # train_data=data.sample(500)
        ###Split the data into train and val just using train test split on label
        all_unique_labels=data['label'].unique()

        ###Take 80% of the labels for train and 20% for val
        train_labels=np.random.choice(all_unique_labels,int(len(all_unique_labels)*0.8),replace=False)
        val_labels=[x for x in all_unique_labels if x not in train_labels]

        ###Get the train and val data
        train_data=data[data['label'].isin(train_labels)]
        val_data=data[data['label'].isin(val_labels)]
    
        return train_data,val_data

    ###Fir val data, we don't want to use the paths in pr - just keep them for tk! 
    ###Drop rows where image_path does not contain tk_title_img_1025_v2



def prep_unlabelled_data(tk_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/ocr_dataframes_multimodal/EffOCR/tk_image_ocr.csv",
                         pr_title_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/ocr_dataframes_multimodal/EffOCR/pr_image_ocr.csv',
                         pr_partner_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/ocr_dataframes_multimodal/EffOCR/partners_image_ocr.csv'):
    """Prep the unlabelled data for training"""
    ##Load the data
    tk_data=pd.read_csv(tk_path)
    pr_title_data=pd.read_csv(pr_title_path)
    pr_partner_data=pd.read_csv(pr_partner_path)

    ###Concat the data 
    data=pd.concat([tk_data,pr_title_data,pr_partner_data])
    print(len(data))

    ###Make a label column (not needed, but plays nicely with dataloader)
    data['label']=1

    ###Drop duplicates
    data=data.drop_duplicates(subset=['image_path','text'])

    ##Drop if na
    data=data.dropna()

    ###Split the data into train and val just using train test split
    train_data,val_data=train_test_split(data,test_size=0.2,random_state=42)



    return train_data,val_data
    




def pretrain_clip(train_loader,model,device,img_loss,text_loss,epoch,optimizer,tokenizer,scheduler=None,epochviz=None):
    print("Pretraining CLIP")
    """REf: https://github.com/openai/CLIP/blob/main/clip/model.py | https://github.com/openai/CLIP/issues/83 """
    # model.to(device)
    model.train()

    loss_list=[]
    for batch_idx, (text, image_data, labels, image_path) in tqdm(enumerate(train_loader)):
        labels = labels.to(device)

        ####Unsquueze the image data
        image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = ja_clip.tokenize(text,tokenizer=tokenizer)

        for key in text_features.keys():
            text_features[key]=text_features[key].to(device)
        
        optimizer.zero_grad()



        model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                attention_mask=text_features["attention_mask"], position_ids=text_features["position_ids"])
        logits_per_image, logits_per_text = model_output["logits_per_image"], model_output["logits_per_text"]
        del model_output

        ###The clip objective asks us to maximize the similarity between the logits of the image and text. Labels aren't needed here.
        ###Giving them a diff label for each image and text pair
        labels=torch.arange((labels.shape[0]))
        labels=labels.to(device)

        loss = (img_loss(logits_per_image, labels) + text_loss(logits_per_text, labels))/2

        loss.backward()

        optimizer.step()

        ##For ReduceLROnPlateau scheduler, we need to pass the loss value
        if scheduler!=None:
            scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(scheduler.get_lr()[0]))
            wandb.log({"train/lr": scheduler.get_lr()[0]})
        wandb.log({"train/loss": loss.item()})

        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(10):
                    image = T.ToPILImage()(INV_NORMALIZE(image_data[i].cpu()))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))

        loss_list.append(loss.item())
    
    ####Mean epoch loss
    mean_epoch_loss=np.mean(loss_list)
    return mean_epoch_loss



def train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,tokenizer,clip_scheduler=None,epochviz=None,mlp_model=None,mlp_optimizer=None,mlp_scheduler=None,freeze_clip=False):
    """A version where we contrastively train pooled clip embeddings"""


    clip_model.train()
    if not mlp_model is None:
        mlp_model.train()


    for batch_idx, (text, image_data, labels, image_path) in tqdm(enumerate(train_loader)):

        labels = labels.to(device)

        ####Unsquueze the image data
        image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = ja_clip.tokenize(text,tokenizer=tokenizer)

        for key in text_features.keys():
            text_features[key]=text_features[key].to(device)
        
        clip_optimizer.zero_grad()
        if not mlp_model is None:
            mlp_optimizer.zero_grad()


        if freeze_clip:
            with torch.no_grad():
                model_output=clip_model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                attention_mask=text_features["attention_mask"], position_ids=text_features["position_ids"])
        
        else:
            model_output=clip_model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                attention_mask=text_features["attention_mask"], position_ids=text_features["position_ids"])
        image_embeds, text_embeds = model_output["image_embeds"], model_output["text_embeds"]
        del model_output

        if args.pooling_type=="mean":
            final_embeds= args.im_wt*image_embeds + (1-args.im_wt)*text_embeds
        elif args.pooling_type=="mlp":
            ###Use an MLP to combine the image and text embeddings
            ###concat the image and text embeddings
            final_embeds=torch.cat([image_embeds,text_embeds],dim=1)
            ###Pass through an MLP
            final_embeds=mlp_model.forward(final_embeds)
        else:
            raise ValueError("Pooling type not supported")

        loss=loss_func(final_embeds,labels)

        loss.backward()

        clip_optimizer.step()
        if not mlp_optimizer is None:
            mlp_optimizer.step()

        ##For ReduceLROnPlateau scheduler, we need to pass the loss value
        if clip_scheduler!=None:
            clip_scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(clip_scheduler.get_lr()[0]))
            wandb.log({"train/clip_lr": clip_scheduler.get_lr()[0]})
        
        if mlp_scheduler!=None:
            mlp_scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(mlp_scheduler.get_lr()[0]))
            wandb.log({"train/mlp_lr": mlp_scheduler.get_lr()[0]})

        wandb.log({"train/loss": loss.item()})

        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(10):
                    image = T.ToPILImage()(INV_NORMALIZE(image_data[i].cpu()))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))


        

def get_image_text_embeddings(data_loader,clip_model,mlp_model,device):
    clip_model.eval()
    if not mlp_model is None:
        mlp_model.eval()

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


            model_output=clip_model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                    attention_mask=text_features["attention_mask"], position_ids=text_features["position_ids"])
            image_embeds, text_embeds = model_output["image_embeds"], model_output["text_embeds"]

            # final_embeds=torch.cat((image_embeds,text_embeds),dim=1)
            ###MEan of the two embeddings
            if args.pooling_type=="mean":
                final_embeds= args.im_wt*image_embeds + (1-args.im_wt)*text_embeds
            elif args.pooling_type=="mlp":
                ###Use an MLP to combine the image and text embeddings
                ###concat the image and text embeddings
                final_embeds=torch.cat([image_embeds,text_embeds],dim=1)
                ###Pass through an MLP
                final_embeds=mlp_model.forward(final_embeds)
                
            # final_embeds=text_embeds
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


def tester_bienc_clip(test_loader,ref_loader,clip_model,mlp_model,split='val',log=True):
    print("Testing using pooled embeddings")

    
    test_embeddings, test_labels, test_text, test_paths = get_image_text_embeddings(test_loader,clip_model,mlp_model, device)
    print("total test embeddings: ",test_embeddings.shape)
    ref_embeddings, ref_labels, ref_text, ref_paths = get_image_text_embeddings(ref_loader,clip_model,mlp_model, device)
    print("total ref embeddings: ",ref_embeddings.shape)
    ###Make an index
    index = faiss.IndexFlatIP(test_embeddings.shape[1])
    index.add(ref_embeddings.cpu().numpy())

    ###Get the nearest neighbours
    D, I = index.search(test_embeddings.cpu().numpy(), 1)


    acc=0
    for i in range(len(test_labels)):
        if test_labels[i]==ref_labels[I[i][0]]:
            acc+=1
    acc=acc/len(test_labels)
    print("CUSTOM ACCURACY: ",acc)


    if log:
        wandb.log({f"{split}/precision_1": acc})

    ###Print a sample of predictions (text)
    for i in range(10):
        print(f"Text: {test_text[i]}")
        print(f"Nearest neighbour: {ref_text[I[i][0]]}")
        print(f"Nearest neighbour label: {ref_labels[I[i][0]]}")
        print(f"Test label: {test_labels[i]}")
        print("")
        print(acc)

    return acc








if __name__ == "__main__":

        ##parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clip_lr", type=float, default=5e-7)
    parser.add_argument("--mlp_lr", type=float, default=5e-5)
    parser.add_argument("--clip_weight_decay",type=float,default=0.001)
    parser.add_argument("--mlp_weight_decay",type=float,default=0.001)
    parser.add_argument("--batch_size",type=int,default=153)
    parser.add_argument("--m",type=int,default=1)
    parser.add_argument("--k",type=int,default=3)
    parser.add_argument("--train_data_type",type=str,default="labelled")
    parser.add_argument("--wandb_name",type=str,default="clip_pretrain_labelled_m1")
    parser.add_argument("--training_type",type=str,default="pretrain")
    parser.add_argument("--supcon_temp",type=float,default=0.1)
    parser.add_argument("--im_wt",type=float,default=0.3)
    parser.add_argument("--pooling_type",type=str,default="mean")
    parser.add_argument("--freeze_clip_epochs",type=int,default=20)
    parser.add_argument("--mlp_layers",type=int,default=3)
    parser.add_argument("--augmented_crops",action="store_true")
    
    parser.add_argument("--train_hardneg",action="store_true")
    parser.add_argument("--checkpoint_path",type=str,default=None)

    args = parser.parse_args()
    
    # def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    clip_model = ja_clip.clip.CLIPModel.from_pretrained("rinna/japanese-clip-vit-b-16",cache_dir="/tmp/japanese_clip")

    ###Load checkpoint
    if args.checkpoint_path is not None:
        clip_model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))
    # model.load_state_dict(torch.load("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/bienc_clip_pretrain_labelled_m3.pt", map_location=torch.device(device)))
    clip_model.to(device)

    if args.pooling_type=="mlp":
        mlp_model=encoders.MLP(2 * 512, 1024, 512, args.mlp_layers, 0.1)
        mlp_model.to(device)
    else:
        mlp_model=None


    tokenizer = ja_clip.load_tokenizer()

    ###DATAparallel

    # model = torch.nn.DataParallel(model)
    # model.to(device)

    if args.train_data_type == "labelled":
        train_data,val_data,full_labelled_data=prep_labelled_data()
    elif args.train_data_type == "unlabelled":
        train_data,val_data=prep_unlabelled_data()
    elif args.train_data_type == "synth":
        train_data,val_data=prep_synth_data()
    elif args.train_data_type == "synth_unlabelled":
        train_data_synth,val_data_synth=prep_synth_data()
        train_data_unlabelled,val_data_unlabelled=prep_unlabelled_data()
        
        print("train_data_unlabelled.shape: ",train_data_unlabelled.shape)
        print("train_data_synth.shape: ",train_data_synth.shape)

        ##GEt unique synth data (by label)
        train_data_synth=train_data_synth.drop_duplicates(subset="label")
        val_data_synth=val_data_synth.drop_duplicates(subset="label")

        ###Sample only same number of rows as unlabelled data from synth data  to keep it a bit balanced
        # train_data_synth=train_data_synth.sample(n=len(train_data_unlabelled),random_state=42)
        # val_data_synth=val_data_synth.sample(n=len(val_data_unlabelled),random_state=42)



        ####Add the synth data to the unlabelled data
        train_data=pd.concat([train_data_synth,train_data_unlabelled])
        val_data=pd.concat([val_data_synth,val_data_unlabelled])

        del train_data_synth,val_data_synth,train_data_unlabelled,val_data_unlabelled
    else:
        raise ValueError("labelled_data must be either labelled, unlabelled or synth")
    
    ###Full labelled data is needed anyway
    _,_,full_labelled_data=prep_labelled_data()

    ###Remove any unnamed columns
    train_data=train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    val_data=val_data.loc[:, ~val_data.columns.str.contains('^Unnamed')]

    ####In the image_path columns, replace /data01/ with 122a7683-fa4b-45dd-9f13-b18cc4f4a187
    train_data['image_path']=train_data['image_path'].apply(lambda x: x.replace("/data01/","/122a7683-fa4b-45dd-9f13-b18cc4f4a187/"))
    val_data['image_path']=val_data['image_path'].apply(lambda x: x.replace("/data01/","/122a7683-fa4b-45dd-9f13-b18cc4f4a187/"))
    
    full_labelled_data['image_path']=full_labelled_data['image_path'].apply(lambda x: x.replace("/data01/","/122a7683-fa4b-45dd-9f13-b18cc4f4a187/"))
    

    ###We wil drop duplicates in the train data if pretraining
    if args.training_type == "pretrain":
        ###Shuffle first
        print("Lenth of train data before dropping duplicates: ",len(train_data))
        train_data=train_data.sample(frac=1,random_state=42)
        train_data=train_data.drop_duplicates(subset=['text'],keep='first')
        print("Lenth of train data after dropping duplicates: ",len(train_data))
    
    
    ###Get test data
    test_data=get_pr_partner_test_data()

    ##Make reference data
    ###Create a reference dataset - for training bienc_clip
    tk_universe_df=get_tk_universe()


    ###Get labels for the reference dataset
    ref_data=tk_universe_df[['image_path','text']]
    ##Merge the reference data with the data to get the labels
    ref_data=pd.merge(ref_data,full_labelled_data[['image_path','label']],on='image_path',how='left')

    ###Wherever the label is nan, set it to -1
    ref_data['label']=ref_data['label'].apply(lambda x: -1 if pd.isna(x) else x)

    ###Make the ref set more tractable. Get 1000 from label that has -1 and keep all the rest!
    small_ref_data=ref_data[ref_data['label']==-1].sample(1).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    med_ref_data=ref_data[ref_data['label']==-1].sample(1000).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    large_ref_data=ref_data[ref_data['label']==-1].sample(7000).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    huge_ref_data=ref_data[ref_data['label']==-1].sample(15000).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    all_tk_ref_data=ref_data[ref_data['label']==-1].append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])


    ##Give more labguage to the 'text'. This is a hack to make the model learn better
    # train_data['text']=train_data['text'].apply(lambda x: "会社名(" + x + ")は次のように書かれています")
    # val_data['text']=val_data['text'].apply(lambda x: "会社名(" + x + ")は次のように書かれています")

    print(len(train_data))
    print(train_data.head())
    ###Create the data datsets
    if args.augmented_crops:
        train_image_transform=create_clip_random_doc_transform()
    else:
        train_image_transform=CLIP_BASE_TRANSFORM
    if args.train_hardneg:
        print("Setting up dataset with hardnegatives")
        dedup_train_data=train_data.drop_duplicates(subset=['label'],keep='first')
        print("Total number of unique labels in train data: ",len(dedup_train_data))
        k_hardneg_df = data_loaders.make_hard_neg_df(dedup_train_data,k=args.k,clip_model=clip_model,mlp_model=mlp_model,device=device,tokenizer=tokenizer,pooling_type=args.pooling_type,im_wt=args.im_wt)
        train_dataset=data_loaders.TextImageDatasetWithHardNegs(train_data,k_hardneg_df,img_transform=  train_image_transform ,text_transform=None,batch_size=126,k=args.k,m=args.m)
        print("Done setting up dataset with hardnegatives")
    else: 
        train_dataset=data_loaders.TextImageDataset(train_data, img_transform=train_image_transform)
    
    val_dataset=data_loaders.TextImageDataset(val_data,img_transform=CLIP_BASE_TRANSFORM)
    small_ref_dataset=data_loaders.TextImageDataset(small_ref_data,img_transform=CLIP_BASE_TRANSFORM)
    med_ref_dataset=data_loaders.TextImageDataset(med_ref_data,img_transform=CLIP_BASE_TRANSFORM)
    large_ref_dataset=data_loaders.TextImageDataset(large_ref_data,img_transform=CLIP_BASE_TRANSFORM)
    huge_ref_dataset=data_loaders.TextImageDataset(huge_ref_data,img_transform=CLIP_BASE_TRANSFORM)
    all_tk_ref_dataset=data_loaders.TextImageDataset(all_tk_ref_data,img_transform=CLIP_BASE_TRANSFORM)
    test_dataset=data_loaders.TextImageDataset(test_data,img_transform=CLIP_BASE_TRANSFORM)

    print(len(train_dataset))


    ###Create the data loaders
    if args.train_data_type == "labelled" or args.train_data_type == "synth":
        
        if args.train_hardneg:
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=126,shuffle=False,num_workers=4)
        else:
            train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=data_loaders.NoReplacementMPerClassSampler(train_dataset, m=args.m,batch_size=args.batch_size,num_passes=1))

        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    elif args.train_data_type == "unlabelled":
        if args.train_hardneg:
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=126,shuffle=False,num_workers=4)
        else:
            train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.train_data_type == "synth_unlabelled":
        if args.train_hardneg:
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=126,shuffle=False,num_workers=16)
        else:
            train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        raise ValueError("labelled_data must be either labelled, unlabelled or synth")

    small_ref_loader=torch.utils.data.DataLoader(small_ref_dataset, batch_size=args.batch_size, shuffle=False)
    med_ref_loader=torch.utils.data.DataLoader(med_ref_dataset, batch_size=args.batch_size, shuffle=False)
    large_ref_loader=torch.utils.data.DataLoader(large_ref_dataset, batch_size=args.batch_size, shuffle=False)
    huge_ref_loader=torch.utils.data.DataLoader(huge_ref_dataset, batch_size=args.batch_size, shuffle=False)
    all_tk_ref_loader=torch.utils.data.DataLoader(all_tk_ref_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ###ADditonally, if training biencoder with synthetic data, create a reference dataset
    if args.train_data_type == "synth":
        # synth_ref_data=pd.concat([train_data,val_data])
        synth_ref_data=pd.concat([val_data])
        ##Shuffle the data
        synth_ref_data=synth_ref_data.sample(frac=1)
        ##Drop duplicates
        synth_ref_data=synth_ref_data.drop_duplicates(subset=['label'])
        synth_ref_dataset=data_loaders.TextImageDataset(synth_ref_data, img_transform=CLIP_BASE_TRANSFORM)
        synth_ref_dataloader=torch.utils.data.DataLoader(synth_ref_dataset, batch_size=args.batch_size, shuffle=False)

        large_synth_ref_data=pd.concat([train_data.sample(20000),val_data])
        large_synth_ref_data=synth_ref_data.sample(frac=1)
        large_synth_ref_data=synth_ref_data.drop_duplicates(subset=['label'])
        large_synth_ref_dataset=data_loaders.TextImageDataset(large_synth_ref_data, img_transform=CLIP_BASE_TRANSFORM)
        large_synth_ref_dataloader=torch.utils.data.DataLoader(large_synth_ref_dataset, batch_size=args.batch_size, shuffle=False)


        val_data=val_data.drop_duplicates(subset=['label'])
        val_dataset=data_loaders.TextImageDataset(val_data, img_transform=CLIP_BASE_TRANSFORM)
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    ###Set up device
    # setup

    img_loss=nn.CrossEntropyLoss()
    text_loss=nn.CrossEntropyLoss()


    ###Optimizer for both clip and mlp
    clip_optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.clip_lr,weight_decay=args.clip_weight_decay, betas=(0.9,0.98),
                    eps=1e-06)
    clip_scheduler = CosineAnnealingWarmRestarts(clip_optimizer, 10, 2)

    if args.pooling_type=="mlp":
        mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=args.mlp_lr,weight_decay=args.mlp_weight_decay, betas=(0.9,0.98),
                        eps=1e-06)
        mlp_scheduler = CosineAnnealingWarmRestarts(mlp_optimizer, 10, 2)
    
    else :
        mlp_optimizer=None
        mlp_scheduler=None



    ###Set up the trainer
    wandb.init(project="multimodal_record_linkage", name=args.wandb_name)
    num_epochs=1000
    start_epoch=0
    best_acc=0
    if args.training_type=="pretrain":
        zero_shot_loss=eval_clip(val_loader,clip_model,tokenizer)

    if args.training_type=="pretrain":


        for epoch in (range(start_epoch, num_epochs+start_epoch)):
            train_loss=pretrain_clip(train_loader,clip_model,device,img_loss,text_loss,epoch,clip_optimizer,tokenizer,clip_scheduler,epochviz=None)
            val_loss=eval_clip(val_loader,clip_model,tokenizer)
            # print("val Accuracy: {}".format(acc))
            # acc=tester_bienc_clip(val_loader,val_loader,model,split="val_small",log=True)
            print("Val loss: {}".format(val_loss))
            print("Train loss: {}".format(train_loss))

            if val_loss<zero_shot_loss:
                zero_shot_loss=val_loss
                torch.save(clip_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",args.wandb_name+".pt"))
                
                print("Model saved at epoch {}".format(epoch))
                print("Path of the saved model: {}".format(os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",args.wandb_name+".pt")))
                print("Path of the saved model: {}".format(os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("epoch_"+str(epoch)+"_"+args.wandb_name+".pt"))))
                print("Val loss: {}".format(val_loss))
                if val_loss<0.1:
                    ###Look at final acc on tk
                    final_acc=tester_bienc_clip(test_loader,small_ref_loader,clip_model,split="test",log=True)
                    print("Final acc on test set: {}".format(final_acc))
            torch.save(clip_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("epoch_"+str(epoch)+args.wandb_name+".pt")))


    elif args.training_type=="train_bienc" and args.train_data_type=="labelled":
        best_acc=tester_bienc_clip(val_loader,small_ref_loader,clip_model,mlp_model,split="val_small",log=True)
        best_acc=tester_bienc_clip(val_loader,med_ref_loader,clip_model,mlp_model,split="val_med",log=True)
        best_acc=tester_bienc_clip(val_loader,large_ref_loader,clip_model,mlp_model,split="val_large",log=True)
        # best_acc=tester_bienc_clip(val_loader,huge_ref_loader,clip_model,mlp_model,split="val_huge",log=True)
        loss_func=losses.SupConLoss(temperature=args.supcon_temp)
        for epoch in (range(start_epoch, num_epochs+start_epoch)):
            if epoch<= args.freeze_clip_epochs:
                if args.pooling_type=="mlp":
                    freeze_clip=True
                else:
                    freeze_clip=False
                epoch_loss=train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,clip_scheduler=clip_scheduler,epochviz=None,tokenizer=tokenizer,mlp_model=mlp_model,mlp_optimizer=mlp_optimizer,mlp_scheduler=mlp_scheduler,freeze_clip=freeze_clip)
            else:
                freeze_clip=False
                epoch_loss=train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,clip_scheduler=clip_scheduler,epochviz=None,tokenizer=tokenizer,mlp_model=mlp_model,mlp_optimizer=mlp_optimizer,mlp_scheduler=mlp_scheduler,freeze_clip=freeze_clip)
            acc=tester_bienc_clip(val_loader,large_ref_loader,clip_model,mlp_model,split="val_large",log=True)
            if acc>best_acc:
                best_acc=acc
                torch.save(clip_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("clip_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt")))
                print("Model saved at epoch {}".format(epoch))
                print("Path of the saved model: {}".format(os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("clip_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt"))))
                if args.pooling_type=="mlp":
                    torch.save(mlp_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("mlp_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt")))
                print("Model saved at epoch {}".format(epoch))
                if acc>0.99:
                    ###Look at final acc on tk
                    final_acc=tester_bienc_clip(test_loader,small_ref_loader,clip_model,mlp_model,split="test",log=True)
            ###SAve at every epoch
            torch.save(clip_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("clip_imwt_"+str(args.im_wt)[2]+"epoch_"+str(epoch)+args.wandb_name+".pt")))
    
    elif args.training_type=="train_bienc" and args.train_data_type!="labelled":
        best_acc=tester_bienc_clip(val_loader,synth_ref_dataloader,clip_model,mlp_model,split="val_small",log=True)
        loss_func=losses.SupConLoss(temperature=args.supcon_temp)
        for epoch in (range(start_epoch, num_epochs+start_epoch)):
            epoch_loss=train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,clip_scheduler=clip_scheduler,epochviz=None,tokenizer=tokenizer,mlp_model=mlp_model,freeze_clip=False)
            acc=tester_bienc_clip(val_loader,large_synth_ref_dataloader,clip_model,mlp_model,split="val_large",log=True)
            if acc>best_acc:
                best_acc=acc
                torch.save(clip_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",args.wandb_name+".pt"))
                print("Model saved at epoch {}".format(epoch))
                print("Path of the saved model: {}".format(os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",args.wandb_name+".pt")))
                if args.pooling_type=="mlp":
                    torch.save(mlp_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("mlp_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt")))
                print("Model saved at epoch {}".format(epoch))
                if acc>0.99:
                    ###Look at final acc on tk
                    final_acc=tester_bienc_clip(test_loader,all_tk_ref_loader,clip_model,mlp_model,split="test",log=True)
            ###SAve at every epoch
            torch.save(clip_model.state_dict(), os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/",("clip_imwt_"+str(args.im_wt)[2]+"epoch_"+str(epoch)+args.wandb_name+".pt")))
    
    else :
        print("Training type not recognised")
        raise ValueError





