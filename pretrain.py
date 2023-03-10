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






def tokenize_sentences(tokenizer,sentences):
    # tokenized_sentences = tokenizer.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return tokenized_sentences


def trainer_knn_cross_modal(image_enc,text_enc, loss_func, device, train_loader, optimizer, epoch, epochviz=None, scheduler=None,tokenizer=None,freeze_vision=True,freeze_text=True):
    """Train by bringing corresponding text and image data together"""

    image_enc=image_enc.to(device)
    text_enc=text_enc.to(device)

    image_enc.train()
    text_enc.model.train()


    for batch_idx, (text, image_data, labels, image_path) in enumerate(train_loader):
            
            labels = labels.to(device)
    
            ####Unsquueze the image data
            image_data = image_data.to(device)
    
            ### text is a tuple of strings, we need to convert it to a tensor
            text=list(text)
            text_features = tokenize_sentences(tokenizer,text)
            ##Send to device. Text features are a dict of tensors
            for key in text_features.keys():
                text_features[key]=text_features[key].to(device)
        
            
            optimizer.zero_grad()
    
            ###Get image embeddings
            if freeze_vision:
                with torch.no_grad():
                    image_cls = image_enc.forward(image_data)
            else:
                image_cls = image_enc.forward(image_data)

            ###Get text embeddings
            if freeze_text:
                with torch.no_grad():
                    text_cls = text_enc.forward(text_features)
            else:
                text_cls = text_enc.forward(text_features)

            ###Extend labels by concating it to itself
            
            labels = torch.cat([labels, labels], dim=0)

            ###Concatenate the image and text embeddings
            embeddings = torch.cat([image_cls, text_cls], dim=0)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            # print("Embeddings shape: {}".format(embeddings.shape))
            # print("Labels shape: {}".format(labels.shape))
            ##Shuffle the embeddings and labels with the same permutation
            # perm = torch.randperm(embeddings.shape[0])
            # embeddings = embeddings[perm]
            # labels = labels[perm]

    
            ##Normalize the embeddings
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



def trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz=None, scheduler=None,tokenizer=None,freeze_vision=True,freeze_text=True):
    model=model.to(device)

    model.train()

    for batch_idx, (text, image_data, labels, image_path) in enumerate(train_loader):

        labels = labels.to(device)

        ####Unsquueze the image data
        image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = tokenize_sentences(tokenizer,text)
         ##Send to device. Text features are a dict of tensors
        for key in text_features.keys():
            text_features[key]=text_features[key].to(device)
       
        
        optimizer.zero_grad()

        if args.fusion_head == "transformer":
            image_cls,text_cls=model.forward(image_data, text_features,freeze_vision=freeze_vision,freeze_text=freeze_text)
            image_cls = torch.nn.functional.normalize(image_cls, p=2, dim=1)
            text_cls = torch.nn.functional.normalize(text_cls, p=2, dim=1)
            embeddings = torch.cat([image_cls, text_cls], dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            embeddings = model.forward(image_data, text_features,freeze_vision=freeze_vision,freeze_text=freeze_text)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        ##Normalize the embeddings
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


def tester_knn(test_set, ref_set, model, accuracy_calculator, split, log=True,tokenizer=None,image_enc=None,text_enc=None):


    if args.fusion_head != "cross_modal":
        model.eval()
        test_embeddings, test_labels, test_text, _ = get_all_embeddings(test_set, model,tokenizer=tokenizer)
        ref_embeddings, ref_labels, ref_text, _ = get_all_embeddings(ref_set, model,tokenizer=tokenizer)
    else:
        image_enc.eval()
        text_enc.model.eval()
        test_embeddings, test_labels, test_text, _ = get_all_cross_modal_embeddings(test_set,image_enc,text_enc,tokenizer=tokenizer)
        ref_embeddings, ref_labels, ref_text, _ = get_all_cross_modal_embeddings(ref_set,image_enc,text_enc,tokenizer=tokenizer)
    
    ###use faiss to calculate precision accuracy
    ##train a faiss index
    index = faiss.IndexFlatIP(ref_embeddings.shape[1])
    index.add(ref_embeddings.cpu().numpy())
    ##Get the top 1 nearest neighbour
    D, I = index.search(test_embeddings.cpu().numpy(), 1)
    ##Get the labels of the nearest neighbour
    nearest_neighbour_labels=ref_labels[I]

    ##Convert nearest neighbour labels to a 1D tensor
    nearest_neighbour_labels=torch.tensor(nearest_neighbour_labels).squeeze()

    # print(nearest_neighbour_labels)
    # print(test_labels)
    # ##Get the precision
    # prec_1=accuracy_calculator.get_accuracy(nearest_neighbour_labels,test_labels)

    ##Custom accuracy test if corresponding labels are same
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


def get_all_embeddings(dataset, model, batch_size=128,tokenizer=None,device=torch.device('cuda')):

    for batch_idx, (text, image_data, labels,image_path) in enumerate(tqdm(dataset)):
        ####Unsquueze the image data
        image_data = image_data.to(device)
        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = tokenize_sentences(tokenizer,text)
         ##Send to device
        text_features = sentence_transformers.util.batch_to_device(text_features,device)
        with torch.no_grad():
            if args.fusion_head == "transformer":
                image_cls,text_cls=model.forward(image_data, text_features)
                image_cls = torch.nn.functional.normalize(image_cls, p=2, dim=1)
                text_cls = torch.nn.functional.normalize(text_cls, p=2, dim=1)
                embeddings = torch.cat([image_cls, text_cls], dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            else:
                embeddings = model(image_data, text_features)
        if batch_idx == 0:
            all_embeddings = embeddings
            all_labels = labels
            all_text=text
            all_paths=image_path
        else:
            all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)
            all_text=all_text+text
            all_paths=all_paths+image_path
    return all_embeddings, all_labels, all_text,all_paths

def get_all_cross_modal_embeddings(dataset, image_enc, text_enc, batch_size=128,tokenizer=None,device=torch.device('cuda')):
    for batch_idx, (text, image_data, labels,image_path) in enumerate(tqdm(dataset)):
        
        ####Unsquueze the image data
        image_data = image_data.to(device)
        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = tokenize_sentences(tokenizer,text)

        ##Send to device
        text_features = sentence_transformers.util.batch_to_device(text_features,device)
        with torch.no_grad():
            image_embeddings = image_enc(image_data)
            text_embeddings = text_enc.forward(text_features)
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

            ###Take mean of the embeddings
            embeddings = (image_embeddings + text_embeddings) / 2
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if batch_idx == 0:
            all_embeddings = embeddings
            all_labels = labels
            all_text=text
            all_paths=image_path
        else:
            all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)
            all_text=all_text+text
            all_paths=all_paths+image_path

    return all_embeddings, all_labels, all_text,all_paths




###Runa as script
if __name__ == "__main__":

    ##parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--freeze_vision", action='store_true')
    parser.add_argument("--freeze_text", action='store_true')
    parser.add_argument("--save_path",type=str)
    parser.add_argument("--fusion_head",type=str,default="mlp")
    parser.add_argument("--pooling_type",type=str,default="mean")
    parser.add_argument("--batch_size",type=int,default=63)
    parser.add_argument("--sbert_text_encoder", action='store_true')

    args = parser.parse_args()


    ####Load the image+text paired data (CSV)
    data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_train_data.csv"
    data=pd.read_csv(data_path)
    print("original size of data: {}".format(len(data)))

    ###drop duplicates if image path and text are the same
    data=data.drop_duplicates(subset=['image_path','text'])

    ###Drop if na
    data=data.dropna()


    print("post processing size of data: {}".format(len(data)))


    ###Get labels for the reference dataset
    ref_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_ref_data.csv"
    ref_data=pd.read_csv(ref_data_path)




    ###Create test data
    test_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/multimodal_synth_test_data.csv"
    test_data=pd.read_csv(test_data_path)


    # train_data=data.sample(500)
    ###Split the data into train and val just using train test split on label
    all_unique_labels=data['label'].unique()

    ###Take 80% of the labels for train and 20% for val
    train_labels=np.random.choice(all_unique_labels,int(len(all_unique_labels)*0.8),replace=False)
    val_labels=[x for x in all_unique_labels if x not in train_labels]

    ###Get the train and val data
    train_data=data[data['label'].isin(train_labels)]
    val_data=data[data['label'].isin(val_labels)]
    ###SAve for inspection
    # train_data.to_csv("train_data.csv")
    # val_data.to_csv("val_data.csv",index=False)

    ##Total unique classes in train data
    print("Total unique classes in train data: {}".format(len(train_data['label'].unique())))

    print(len(val_data),"val data without dropping ")
    print(val_data.head(1))
    ###print columns
    print(val_data.columns)
    ###print value of each variable for first row
    print(val_data.iloc[0])


    ###For val_data, sample only one image per label to reduce effort for validation - drop duplicates by label randomly
    val_data=val_data.drop_duplicates(subset=['label'],keep='last').reset_index(drop=True)

 

    print(len(val_data),"val data after dropping ")

    val_data.to_csv("val_data_nodup.csv",index=False)

    print(val_data.head(1))
    print(val_data.columns)
    print(val_data.iloc[0])


    ###Create the data datsets
    train_dataset=data_loaders.TextImageDataset(train_data, img_transform=BASE_TRANSFORM) #create_random_no_aug_doc_transform()
    val_dataset=data_loaders.TextImageDataset(val_data,img_transform=BASE_TRANSFORM)
    ref_dataset=data_loaders.TextImageDataset(ref_data,img_transform=BASE_TRANSFORM)


    ###Create the data loaders
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=data_loaders.NoReplacementMPerClassSampler(train_dataset, m=3,batch_size=args.batch_size,num_passes=1))
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=300, shuffle=True)
    ref_loader=torch.utils.data.DataLoader(ref_dataset, batch_size=300, shuffle=True)


    ###Set up device
    # setup

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    ###image encoder
    auto_timm_model="vit_base_patch16_224.dino"
    image_model_cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/vit_word_nohn_japan_center_places_20000_finetuned/enc_best_e.pth"
    image_encoder = encoders.AutoVisionEncoderFactory("timm", auto_timm_model)
    image_enc=image_encoder.load(image_model_cp_path)
    image_enc=image_enc.to(device)
    ###text encoder
    trained_sbert_model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/language_model_linkage/output/2023-02-14_20-02-05/800"
    
    if args.sbert_text_encoder:
        text_encoder = encoders.SBERTEncoder(trained_sbert_model_path)
        sbert_model_for_tokenizer=SentenceTransformer(trained_sbert_model_path)
        tokenizer=sbert_model_for_tokenizer.tokenizer
        text_encoder=text_encoder.to(device)
    else:
        text_encoder = encoders.BertEncoder("cl-tohoku/bert-base-japanese")
        # text_encoder=text_encoder.load(trained_sbert_model_path)
        text_encoder=text_encoder.to(device)
        tokenizer=text_encoder.tokenizer

    ###Create the multimodal model

    if args.fusion_head=="mlp":
        model = encoders.MultiModalEncoder(image_enc, text_encoder,num_layers=1)
    elif args.fusion_head=="pooling" :
        model = encoders.MultiModalPoolingEncoder(image_enc, text_encoder,args.pooling_type)
    elif args.fusion_head=="transformer":
        model = encoders.MultiModalTransformerEncoder(image_enc, text_encoder,args.pooling_type)
    else:
        print("Cross modal")
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



    ###Create the loss function
    loss_func = losses.SupConLoss(temperature = 0.1) 

    ##Optimizer
    if args.fusion_head in ["mlp","transformer","pooling"]:
        model=model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    else : 
        print("Using cross modal loss")
        optimizer = AdamW(list(text_encoder.model.parameters())+list(image_enc.parameters()), lr=args.lr, weight_decay=0.05)

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
    if args.fusion_head in ["mlp","transformer","pooling"]:
        best_acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, "val",tokenizer=tokenizer)
    else:
        print("Using cross modal loss")
        model=None
        best_acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, "val",tokenizer=tokenizer,image_enc=image_enc,text_enc=text_encoder)

    for epoch in (range(start_epoch, num_epochs+start_epoch)):

        if args.fusion_head=="pooling":
            trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=args.freeze_vision,freeze_text=args.freeze_text)
            acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer)


        elif epoch < 5 and args.fusion_head=="transformer" or args.fusion_head=="mlp": ##Warm up the transformer
            freeze_vision=True
            freeze_text=True
            trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=freeze_vision,freeze_text=freeze_text)
            acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer)
        
        elif epoch >= 5 and args.fusion_head=="transformer" or args.fusion_head=="mlp" and epoch <= 10:
            freeze_vision=False
            freeze_text=True
            trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=freeze_vision,freeze_text=freeze_text)
            acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer)

        elif epoch > 10 and args.fusion_head=="transformer" or args.fusion_head=="mlp":
            freeze_vision=False
            freeze_text=False
            trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=freeze_vision,freeze_text=freeze_text)
            acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer)

        else : 
            print("Using cross modal loss")
            if epoch < 5:
                freeze_text=True
                freeze_vision=False
                trainer_knn_cross_modal(image_enc,text_encoder, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=freeze_vision,freeze_text=freeze_text)
                acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer,image_enc=image_enc,text_enc=text_encoder)
            elif epoch >= 5 and epoch <= 10:
                freeze_text=False
                freeze_vision=True
                trainer_knn_cross_modal(image_enc,text_encoder, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=freeze_vision,freeze_text=freeze_text)
                acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer,image_enc=image_enc,text_enc=text_encoder)

            else:
                freeze_text=False
                freeze_vision=False
                trainer_knn_cross_modal(image_enc,text_encoder, loss_func, device, train_loader, optimizer, epoch, epochviz="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/",scheduler=scheduler,tokenizer=tokenizer,freeze_vision=freeze_vision,freeze_text=freeze_text)
                acc=tester_knn(val_loader, ref_loader, model, accuracy_calculator, f"val",tokenizer=tokenizer,image_enc=image_enc,text_enc=text_encoder)

        if acc>= best_acc:
            best_acc=acc
            torch.save(model.state_dict(), args.save_path)
            print("saved best model")
    

        





