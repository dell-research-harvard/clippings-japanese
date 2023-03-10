####Training a multimodal model

import torch
import torch.nn as nn

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

import math
import json
import argparse

from sklearn.model_selection import train_test_split




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



def tokenize_sentences(sbert_model,sentences):
    tokenised_sentences = sbert_model.tokenize(sentences)
    return tokenised_sentences


def trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz=None, scheduler=None,sbert_model=None,freeze_vision=True,freeze_text=True):
    # model=model.to(device)

    model.train()

    for batch_idx, (text, image_data, labels) in enumerate(train_loader):

        # labels = labels.to(device)

        ####Unsquueze the image data
        # image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = tokenize_sentences(sbert_model,text)
         ##Send to device. Text features are a dict of tensors
        # for key in text_features.keys():
        #     text_features[key]=text_features[key].to(device)
       
        
        optimizer.zero_grad()


        embeddings = model(image_data, text_features,freeze_vision=freeze_vision,freeze_text=freeze_text)
        ##Normalize the embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        loss = loss_func(embeddings, labels)
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


def tester_knn(test_set, ref_set, model, accuracy_calculator, split, log=True,sbert_model_for_tokenizer=None):

    model.eval()

  
    test_embeddings, test_labels, test_text, _ = get_all_embeddings(test_set, model,sbert_model=sbert_model_for_tokenizer)
    ref_embeddings, ref_labels, ref_text, _ = get_all_embeddings(ref_set, model,sbert_model=sbert_model_for_tokenizer)
    
    ###use faiss to calculate precision accuracy
    ##train a faiss index
    index = faiss.IndexFlatIP(ref_embeddings.shape[1])
    index.add(ref_embeddings.cpu().numpy())
    ##Get the top 1 nearest neighbour
    D, I = index.search(test_embeddings.cpu().numpy(), 1)
    ##Get the labels of the nearest neighbour
    nearest_neighbour_labels=ref_labels[I]


    ##Get the precision
    prec_1=accuracy_calculator.get_accuracy(nearest_neighbour_labels,test_labels)
    if log:
        wandb.log({f"{split}/precision_1": prec_1})

    ###Print a sample of predictions (text)
    for i in range(10):
        print(f"Text: {test_text[i]}")
        print(f"Nearest neighbour: {ref_text[I[i][0]]}")
        print(f"Nearest neighbour label: {ref_labels[I[i][0]]}")
        print(f"Test label: {test_labels[i]}")
        print("")
        print(prec_1)
    return prec_1


def get_all_embeddings(dataset, model, batch_size=128,sbert_model=None,device=torch.device('cuda')):

    for batch_idx, (text, image_data, labels,image_path) in enumerate(tqdm(dataset)):
        ####Unsquueze the image data
        # image_data = image_data.to(device)
        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = tokenize_sentences(sbert_model,text)
         ##Send to device
        text_features = sentence_transformers.util.batch_to_device(text_features,device)
        with torch.no_grad():
            embeddings = model(image_data, text_features)
        if batch_idx == 0:
            all_embeddings = embeddings
            all_labels = labels
            all_text=text
            all_paths=image_path
        else:
            all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_text=all_text+text
            all_paths=all_paths+image_path
    return all_embeddings, all_labels, all_text,all_paths



###Runa as script
if __name__ == "__main__":

    ##parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--freeze_vision", type=bool, default=False)
    parser.add_argument("--freeze_text", type=bool, default=False)
    parser.add_argument("--save_path",type=str)

    args = parser.parse_args()


    ####Load the image+text paired data (CSV)
    data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_train.csv"
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

    ###Create a reference dataset
    tk_universe_df=get_tk_universe()

    ###Get labels for the reference dataset
    ref_data=tk_universe_df[['image_path','text']]
    ##Merge the reference data with the data to get the labels
    ref_data=pd.merge(ref_data,data[['image_path','label']],on='image_path',how='left')

    ###Wherever the label is nan, set it to -1
    ref_data['label']=ref_data['label'].apply(lambda x: -1 if pd.isna(x) else x)

    ###Make the ref set more tractable. Get 1000 from label that has -1 and keep all the rest!
    small_ref_data=ref_data[ref_data['label']==-1].sample(1).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    med_ref_data=ref_data[ref_data['label']==-1].sample(1000).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    large_ref_data=ref_data[ref_data['label']==-1].sample(10000).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    huge_ref_data=ref_data[ref_data['label']==-1].sample(20000).append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])
    all_ref_data=ref_data[ref_data['label']==-1].append(ref_data[ref_data['label']!=-1]).drop_duplicates(subset=['image_path','text'])

    ###Create test data


    # train_data=data.sample(500)
    ###Split the data into train and val just using train test split on label
    all_unique_labels=data['label'].unique()

    ###Take 80% of the labels for train and 20% for val
    train_labels=np.random.choice(all_unique_labels,int(len(all_unique_labels)*0.8),replace=False)
    val_labels=[x for x in all_unique_labels if x not in train_labels]

    ###Get the train and val data
    train_data=data[data['label'].isin(train_labels)]
    val_data=data[data['label'].isin(val_labels)]    

    ###Fir val data, we don't want to use the paths in pr - just keep them for tk! 
    ###Drop rows where image_path does not contain tk_title_img_1025_v2
    val_data=val_data[~val_data['image_path'].str.contains("tk_title_img_1025_v2")]


    ###Create the data datsets
    train_dataset=data_loaders.TextImageDataset(train_data, img_transform=create_random_no_aug_doc_transform())
    val_dataset=data_loaders.TextImageDataset(val_data,img_transform=create_random_no_aug_doc_transform())
    small_ref_dataset=data_loaders.TextImageDataset(small_ref_data,img_transform=BASE_TRANSFORM)
    med_ref_dataset=data_loaders.TextImageDataset(med_ref_data,img_transform=BASE_TRANSFORM)
    large_ref_dataset=data_loaders.TextImageDataset(large_ref_data,img_transform=BASE_TRANSFORM)
    huge_ref_dataset=data_loaders.TextImageDataset(huge_ref_data,img_transform=BASE_TRANSFORM)
    all_ref_dataset=data_loaders.TextImageDataset(all_ref_data,img_transform=BASE_TRANSFORM)

    ###Create the data loaders
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=126,sampler=data_loaders.NoReplacementMPerClassSampler(train_dataset, m=3,batch_size=126,num_passes=1))
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=126, shuffle=True)
    small_ref_loader=torch.utils.data.DataLoader(small_ref_dataset, batch_size=126, shuffle=True)
    med_ref_loader=torch.utils.data.DataLoader(med_ref_dataset, batch_size=126, shuffle=True)
    large_ref_loader=torch.utils.data.DataLoader(large_ref_dataset, batch_size=126, shuffle=True)
    huge_ref_loader=torch.utils.data.DataLoader(huge_ref_dataset, batch_size=126, shuffle=True)
    all_ref_loader=torch.utils.data.DataLoader(all_ref_dataset, batch_size=126, shuffle=True)

    ###Set up device
    # setup

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    ###image encoder
    auto_timm_model="vit_base_patch16_224.dino"
    image_model_cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/vit_word_nohn_japan_center_places_20000_finetuned/enc_best_e.pth"
    image_encoder = encoders.AutoVisionEncoderFactory("timm", auto_timm_model)
    image_enc=image_encoder.load(image_model_cp_path)
    # image_enc=image_enc.to(device)
    ###text encoder
    trained_sbert_model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/language_model_linkage/output/2023-02-14_20-02-05/800"
    text_encoder = encoders.SBERTEncoder(trained_sbert_model_path)
    sbert_model_for_tokenizer=SentenceTransformer(trained_sbert_model_path)
    # text_encoder=text_encoder.to(device)

    ###Create the multimodal model
    model = encoders.MultiModalEncoder(image_enc, text_encoder,num_layers=1)
    # model = encoders.MultiModalPoolingEncoder(image_enc, text_encoder,"mean")
    # model = encoders.MultiModalTransformerEncoder(image_enc, text_encoder,"mean")

    ###Fine tune
    fine_tune=True
    # if fine_tune:
    #     ###Load the pretrained model state dict
    #     pretrained_model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/enc_best_e_only_text_frozen_mlp.pth"
    #     model.load_state_dict(torch.load(pretrained_model_path))


    # del text_encoder
    # del image_enc

    # data parallelism

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        datapara = True
        text_encoder=nn.DataParallel(text_encoder)
        image_enc=nn.DataParallel(image_enc)
        model = nn.DataParallel(model)
        # sbert_model_for_tokenizer = nn.DataParallel(sbert_model_for_tokenizer)
    else:
        datapara = False

    model=model.to(device)


    ###Create the loss function
    loss_func = losses.SupConLoss(temperature = 0.1) 

    ##Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    ###Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 2)
    ##Scheduler that decays the learning rate by a factor after a warmup period
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # scheduler=CustomScheduler(optimizer, eta_min=0.0001, T_max=200, last_epoch=-1)

    ##Init wandb
    wandb.init(project="multimodal_record_linkage", name="test")
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
    num_epochs=args.epochs
    start_epoch=0
    ##Train!
    ###get zero shot accuracy
    acc_0=tester_knn(val_loader, small_ref_loader, model, accuracy_calculator, "val",sbert_model_for_tokenizer=sbert_model_for_tokenizer)["precision_at_1"]
    acc_1=0.9
    acc_2=0.9
    acc_3=0.9
    acc_4=0.9
    # best_acc=0.9
    loader_list=[small_ref_loader,med_ref_loader,large_ref_loader,huge_ref_loader,all_ref_loader]
    loader_list_index=0

    for epoch in range(start_epoch, num_epochs+start_epoch):
        trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,sbert_model=sbert_model_for_tokenizer,freeze_vision=args.freeze_vision,freeze_text=args.freeze_text)
        acc=tester_knn(val_loader, loader_list[loader_list_index], model, accuracy_calculator, f"val_{loader_list_index}",sbert_model_for_tokenizer=sbert_model_for_tokenizer)["precision_at_1"]
        if loader_list_index==0 and acc>=acc_0:
            acc_0=acc
            torch.save(model.state_dict(), args.save_path.split(".")[0]+"_0.pth")
            print("Saved best model")
            loader_list_index=loader_list_index+1         
            print("Switching to largest ref loader: ",loader_list_index)            
       
        elif loader_list_index==1 and acc>=acc_1:
            acc_1=acc
            torch.save(model.state_dict(), args.save_path.split(".")[0]+"_1.pth")
            print("Saved best model")
            loader_list_index=loader_list_index+1   
            print("Switching to largest ref loader: ",loader_list_index)            
             
     
        elif loader_list_index==2 and acc>=acc_2:
            acc_2=acc
            torch.save(model.state_dict(), args.save_path.split(".")[0]+"_2.pth")
            print("Saved best model")     
            loader_list_index=loader_list_index+1     
            print("Switching to largest ref loader: ",loader_list_index)            
           

        elif loader_list_index==3 and acc>=acc_3:
            acc_3=acc
            torch.save(model.state_dict(), args.save_path.split(".")[0]+"_3.pth")
            print("Saved best model")   
            loader_list_index=loader_list_index+1    
            print("Switching to largest ref loader: ",loader_list_index)            
  
        elif loader_list_index==4 and acc>=acc_4:
            acc_4=acc
            torch.save(model.state_dict(), args.save_path.split(".")[0]+"_4.pth")
            print("Saved best model")

        else : 
            pass

        if loader_list_index>0 and acc<0.93:
            loader_list_index=loader_list_index-1
            print("Switching to smaller ref loader: ",loader_list_index)
             
        
        
    

        





