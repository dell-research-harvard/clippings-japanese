
### Encode all target images and source images using ViT base and find nearest neighbors using FAISS
import timm
import json
import os
import sys
import numpy as np
import faiss
import torch
import torchvision 
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel
import pandas as pd
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import os
import hashlib
import shutil
from timm.models import load_state_dict
import argparse
import wandb
from glob import glob
from matplotlib import pyplot as plt
from models.encoders import *

from copy import deepcopy
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')
import cv2
from collections import defaultdict
##Viz attention  masks
from datetime import datetime

from utils.matched_accuracy import calculate_matched_accuracy
from utils.nomatch_accuracy import calculate_nomatch_accuracy

###Collate fn
def diff_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
###Custom transforms

class MedianPad:
    def __init__(self, override=None):

        self.override = override

    def __call__(self, image):

        ##Convert to RGB 
        image = image.convert("RGB") if isinstance(image, Image.Image) else image
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        pad_x, pad_y = [max_side - s for s in image.size]
        padding = (0, 0, pad_x, pad_y)
        padding = (round((10+pad_x)/2), round((5+pad_y)/2), round((10+pad_x)/2), round((5+pad_y)/2)) ##Added some extra to avoid info on the long edge

        imgarray = np.array(image)
        h, w , c= imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        return transforms.Pad(padding, fill=medval if self.override is None else self.override)(image)

def diff_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

BASE_TRANSFORM = transforms.Compose([
        MedianPad(override=(255,255,255)),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

NO_PAD_NO_RESIZE_TRANSFORM = transforms.Compose([
    ##To rgb
        transforms.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
        transforms.ToTensor(),
        transforms.Resize(224,max_size=225),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

def get_embedding_list(image_folder, timm_pretrained_model=None,trained_pth=None, device="cuda", batch_size=16,transform_type='base',xcit=False):
    """
    Encode all images in a folder using ViT base
    
    """
    if xcit:
        ##Load Xcit model for variable size images
        encoder = XcitEncoder
        model = encoder()
    
    else:
        model = timm.create_model(timm_pretrained_model, num_classes=0, pretrained=True,img_size=224)
    
    if trained_pth!=None:
        if xcit:
            model.load_state_dict(clean_checkpoint(trained_pth,clean_net=False))
        else:
            model.load_state_dict(clean_checkpoint(trained_pth))
    model.to(device)
    model.eval()
    
    if transform_type=='base':
        transform = transforms.Compose([transforms.Resize(248,
        interpolation= transforms.InterpolationMode.BICUBIC ),
        transforms.CenterCrop(size=(224, 224))  ,
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])
        ])
    if transform_type=='no_pad_no_resize':
        transform = NO_PAD_NO_RESIZE_TRANSFORM
    else:
        transform=BASE_TRANSFORM

    ##Create im loader
    if transform_type=='no_pad_no_resize':
        image_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(image_folder, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=64,
            collate_fn=diff_size_collate
        )
    else:
        image_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(image_folder, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=64
        )
  
    # Get image embeddings
    print("Embedding...")
    image_embeddings = []


    for images, _ in tqdm(image_loader):

        images = [image.to(device) for image in images]  if transform_type=='no_pad_no_resize' else images.to(device)
        i=0
        with torch.no_grad(): #Get last hidden state
            if transform_type=='no_pad_no_resize':
                for image in images:
                    ##Save the image as png
                    check_image = image.permute(1, 2, 0)
                    check_image = check_image.cpu().numpy()
                    check_image = check_image * 255
                    check_image = check_image.astype(np.uint8)
                    check_image = Image.fromarray(check_image)
                    check_image.save(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/transform_checks/{i}.png')
                    emb = model(image.unsqueeze(0)).squeeze(0).cpu().numpy()
                    emb = emb / np.linalg.norm(emb)
                    emb=deepcopy(emb)
                    image_embeddings.append(emb)
                    del emb
                    i=i+1

                # embeddings = torch.stack(out_emb, dim=0)
            else:
                embedding=model(images).cpu().numpy()
                ##Normalize the embeddings
                embedding = embedding / np.linalg.norm(embedding)
                image_embeddings.append(embedding)

    ####Chunk the imaeges to avoid memory issues
    image_paths=[x[0] for x in image_loader.dataset.imgs]
    print(len(image_embeddings))
    print(image_embeddings[0].shape)
    if transform_type!='no_pad_no_resize':
        image_embeddings = np.concatenate(image_embeddings)
    print(image_embeddings[0].shape)
    print(len(image_embeddings))

    return image_embeddings, image_paths

def get_nearest_neighbors(source_embeddings, target_embeddings, top_k=5):
    """
    Find nearest neighbors using FAISS
    """
    # Create index
    ##Normalise the embeddings
    source_embeddings = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(source_embeddings[0].shape[0])
    print(target_embeddings.shape)
    index.add(target_embeddings)

    # Find nearest neighbors
    distances, indices = index.search(source_embeddings, top_k)

    print("Accuracy: ", np.mean([indices[i][0] == i for i in range(len(indices))]))

    return distances, indices

def make_best_match_df(source_paths,target_paths,indices):
    """
    Make a dataframe of the best matches
    """
    df = pd.DataFrame(columns=["source", "target"])
    for i, source_path in enumerate(source_paths):
        target_path = target_paths[indices[i][0]]
        df = df.append({"source": source_path, "target": target_path}, ignore_index=True)
    return df

def make_topk_match_df(source_paths,target_paths,indices,distances,k):
    """
    Make a dataframe of the top k matches
    """
    df = pd.DataFrame(columns=["source", "target"])
    for i, source_path in enumerate(source_paths):
        for j in range(k):
            target_path = target_paths[indices[i][j]]
            distance=distances[i][j]
            df = df.append({"source": source_path, "target": target_path,"distance":distance}, ignore_index=True)
    return df


def fast_match_topk_dict(source_paths,target_paths,indices,distances,k,og_source_path=None,og_target_path=None):
    """
    Make a dataframe of the top k matches very quickly
    """

    ###The index of indices is the index of the source image. Each element in Indices is a list of the indices of the top k matches in target paths
    ###Convert all indices in Indices to list of target paths so it becomes a list of list of target paths
    indices_paths=[list(map(lambda x: target_paths[x],indices[i])) for i in tqdm(range(len(indices)))]
        
    ###round distances to 4 decimal places and convert them to string
    distances=[[str(round(x,4)) for x in distances[i]] for i in tqdm(range(len(distances)))]
    
    ###MAke a dict with source : {targets:[target1...targetk],distances:[dist1...distk]}
    match_dict=defaultdict(dict)
    for i in tqdm(range(len(source_paths))):
        match_dict[source_paths[i]]={"targets":indices_paths[i],"distances":distances[i]}
    
    return match_dict

def viz_matches(match_df,n=10,save_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/",lang_code="TK"):
    """
    Visualize the matches
    """

    fig, ax = plt.subplots(n, 2, figsize=(10, 10))
    for i in range(n):
        source = Image.open(match_df.iloc[i]["source"])
        target = Image.open(match_df.iloc[i]["target"])
        ax[i][0].imshow(source)
        ax[i][1].imshow(target)
        ax[i][0].set_title("Source")
        ax[i][1].set_title("Target")
        ax[i][0].axis("off")
        ax[i][1].axis("off")
    
    ##Save the figure
    plt.savefig(save_dir+"_"+lang_code+"_"+f"matches_result.png",dpi=600)

def check_match_accuracy(origin_df,result_df,results_dir,lang_code="TK"):
    """
    Check the accuracy of the matches
    """
    print(len(origin_df))
    print(len(result_df))
    ##get only last part of the path for both source and target and for both dataframes
    origin_df["source_fname"]=origin_df["source"].apply(lambda x: os.path.basename(x))
    origin_df["target_fname"]=origin_df["target"].apply(lambda x: os.path.basename(x))
    result_df["source_fname"]=result_df["source"].apply(lambda x: os.path.basename(x))
    result_df["target_fname"]=result_df["target"].apply(lambda x: os.path.basename(x))
    
    correct = 0
    for img_name in origin_df["source_fname"].unique():
        origin_target = origin_df[origin_df["source_fname"] == img_name]["target_fname"].values[0]
        result_target = result_df[(result_df["source_fname"]) == img_name]["target_fname"].values[0]
        if (origin_target) == (result_target):
            correct += 1
    print("Accuracy: ", correct / len(origin_df))
    ###Visualise correct matches
    counter=0
    fig, ax = plt.subplots(10, 2, figsize=(10, 10),gridspec_kw={'width_ratios': [1, 1]})
    for i,img_name in enumerate(origin_df["source_fname"].unique()):
        origin_target = origin_df[origin_df["source_fname"] == img_name]["target_fname"].values[0]
        result_target = result_df[(result_df["source_fname"]) == img_name]["target_fname"].values[0]
        if origin_target != result_target:
            continue
        source = Image.open(result_df[(result_df["source_fname"]) == img_name]["source"].values[0])
        target = Image.open(result_df[(result_df["source_fname"]) == img_name]["target"].values[0])

        ##Add some space between the subplots
        ax[counter][0].imshow(source)
        ##Increase the size of the target image - width of target = height of source with fixed aspect ratio
        # target = target.resize((source.size[1],int(target.size[1]*source.size[1]/target.size[0])))
        ax[counter][1].imshow(target)
        ax[counter][0].set_title("Source")
        ax[counter][1].set_title("Target")
        ax[counter][0].axis("off")
        ax[counter][1].axis("off")
        counter+=1
        if counter==10:
            break
            
    ###Add some space between the subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.savefig(results_dir+"_"+lang_code+"_"+f"correct_matches_result.png",dpi=600)

        ###Visualise incorrect matches
    counter=0
    fig, ax = plt.subplots(10, 3, figsize=(10, 10),gridspec_kw={'width_ratios': [1, 1, 1]})
    for i,img_name in enumerate(origin_df["source_fname"].unique()):
        origin_target = origin_df[origin_df["source_fname"] == img_name]["target_fname"].values[0]
        result_target = result_df[(result_df["source_fname"]) == img_name]["target_fname"].values[0]
        if origin_target == result_target:
            continue
        source = Image.open(result_df[(result_df["source_fname"]) == img_name]["source"].values[0])
        target = Image.open(result_df[(result_df["source_fname"]) == img_name]["target"].values[0])
        origin_target = Image.open(origin_df[origin_df["source_fname"] == img_name]["target"].values[0])

        ax[counter][0].imshow(source)
        ##Increase the size of the target image - width of target = height of source with fixed aspect ratio
        ax[counter][1].imshow(target)
        ax[counter][2].imshow(origin_target)
        ax[counter][0].set_title("Source")
        ax[counter][1].set_title("Target")
        ax[counter][2].set_title("Ground Truth")
        ax[counter][0].axis("off")
        ax[counter][1].axis("off")
        ax[counter][2].axis("off")

        counter+=1
        if counter==10:
            break
    
    ###Add some space between the subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(results_dir+"_"+lang_code+"_"+f"incorrect_matches_result.png",dpi=600)
            
def check_topk_match_accuracy(origin_df,result_df,k,results_dir,lang_code="TK"):
    """
    Check the topk accuracy of the result_df
    """
    ##get only last part of the path for both source and target and for both dataframes
    origin_df["source_fname"]=origin_df["source"].apply(lambda x: os.path.basename(x))
    origin_df["target_fname"]=origin_df["target"].apply(lambda x: os.path.basename(x))
    result_df["source_fname"]=result_df["source"].apply(lambda x: os.path.basename(x))
    result_df["target_fname"]=result_df["target"].apply(lambda x: os.path.basename(x))
    ##Check topk accuracy
    correct = 0
    for img_name in origin_df["source_fname"].unique():
        origin_target = origin_df[origin_df["source_fname"] == img_name]["target_fname"].values[0]
        result_target = result_df[(result_df["source_fname"]) == img_name]["target_fname"].values[:k]
        if (origin_target) in (result_target):
            correct += 1
    
    top_k_accuracy=correct / len(origin_df)
    print(f"Top{k} Accuracy: ", correct / len(origin_df))
    ##Visualise correct matches (match exists in topk)
    counter=0
    fig, ax = plt.subplots(10, k+1, figsize=(10, 10))
    for i,img_name in enumerate(origin_df["source_fname"].unique()):
        origin_target = origin_df[origin_df["source_fname"] == img_name]["target_fname"].values[0]
        result_target = result_df[(result_df["source_fname"]) == img_name]["target_fname"].values[:k]
        if origin_target not in result_target:
            continue
        source = Image.open(result_df[(result_df["source_fname"]) == img_name]["source"].values[0])
        target_imlist = [Image.open(result_df[(result_df["source_fname"]) == img_name]["target"].values[i]) for i in range(k)]
        dist_list=[result_df[(result_df["source_fname"]) == img_name]["distance"].values[i] for i in range(k)]

        ax[counter][0].imshow(source)

        for j,target in enumerate(target_imlist):
            ax[counter][j+1].imshow(target)
            ax[counter][j+1].axis("off")
            ax[counter][j+1].set_title(f"{j+1}: {dist_list[j]:.4f}")
        ax[counter][0].set_title("Source")
        ax[counter][0].axis("off")
        counter+=1

        if counter==10:
            break

    ###Add some space between the subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.savefig(results_dir+"_"+lang_code+"_"+f"correct_matches_result_top{k}.png",dpi=600)


    ##Visualise incorrect matches (match does not exist in topk)
    counter=0
    fig_counter=0
    fig, ax = plt.subplots(10, k+2, figsize=(10, 10))
    print(len(origin_df["source_fname"].unique()))
    for i,img_name in enumerate(origin_df["source_fname"].unique()):
        origin_target = origin_df[origin_df["source_fname"] == img_name]["target_fname"].values[0]
        result_target = result_df[(result_df["source_fname"]) == img_name]["target_fname"].values[:k]
        if origin_target not in result_target:
            
            source = Image.open(result_df[(result_df["source_fname"]) == img_name]["source"].values[0])
            target_imlist = [Image.open(result_df[(result_df["source_fname"]) == img_name]["target"].values[i]) for i in range(k)]
            dist_list=[result_df[(result_df["source_fname"]) == img_name]["distance"].values[i] for i in range(k)]

            ax[counter][0].imshow(source)

            for j,target in enumerate(target_imlist):
                ax[counter][j+1].imshow(target)
                ax[counter][j+1].axis("off")
                ax[counter][j+1].set_title(f"{j+1}: {dist_list[j]:.4f}")
            ax[counter][0].set_title("Source")
            ax[counter][0].axis("off")
            ax[counter][k+1].imshow(Image.open(origin_df[origin_df["source_fname"] == img_name]["target"].values[0]))
            ax[counter][k+1].set_title("Ground Truth")
            ax[counter][k+1].axis("off")
            counter+=1
            

            if counter==10:
                plt.subplots_adjust(wspace=0.5, hspace=0.5)
                plt.savefig(results_dir+"_"+lang_code+"_"+f"incorrect_matches_result_top{k}_{fig_counter}.png",dpi=600)
                fig_counter+=1
                counter=0
                fig, ax = plt.subplots(10, k+2, figsize=(10, 10))
            
            elif counter<=10 and i==233:
                plt.subplots_adjust(wspace=0.5, hspace=0.5)
                plt.savefig(results_dir+"_"+lang_code+"_"+f"incorrect_matches_result_top{k}_{fig_counter}.png",dpi=600)
                fig_counter+=1
                counter=0
                fig, ax = plt.subplots(10, k+2, figsize=(10, 10))

    print((len(origin_df["source_fname"].unique())-1))
    print(top_k_accuracy)
    return(top_k_accuracy)

def get_file_name(path):
    """
    Get file name from path
    """
    fname= os.path.splitext(os.path.basename(path))[0]

    return fname

####CLEAN model cp

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
            
            name = k[4:] if (k.startswith('net.') and clean_net==True) else k
            new_state_dict[name] = v
        print("=> Loaded state_dict from '{}'".format(checkpoint))
        return new_state_dict
    else:
        print("Error: Checkpoint ({}) doesn't exist".format(checkpoint))
        return ''


def main(root_folder, model, trained_model_path , lang_code,wandb_log=False,transform_type="base",xcit=False,recopy=True):
    ##Create root folder if it doesn't exist
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

      ##EMpty lang folder if it exists
    lang_folder=os.path.join(root_folder,lang_code)
    
    if recopy:
        if os.path.exists(lang_folder): ##non-empty folder! - use rm -r
            os.system("rm -r " + lang_folder)

        
    if not os.path.exists(lang_folder):
        os.mkdir(lang_folder)

    ##Create a folder for source images
    source_folder=os.path.join(lang_folder,"source")
    if not os.path.exists(source_folder):
        os.mkdir(source_folder)
    
    ##Create a folder for target images
    target_folder=os.path.join(lang_folder,"target")
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    source_image_paths=glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/pr_partner_crop_36673/*")
    source_root="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/pr_partner_crop_36673/"
    if lang_code=="TK":
        target_root="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/tk_title_crop_68352/"
        target_image_paths=glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/tk_title_crop_68352/*")
    else:
        target_root="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/pr_title_crop_6725/"
        target_image_paths=glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/record_linkage_clean_dataset/pr_title_crop_6725/*")

    if recopy:
        print("Copying files...")

        for i in tqdm(range(len(source_image_paths))):
            os.system("cp "+"\"" +source_image_paths[i]+ "\"" + " " +  source_folder)
            
        for i in tqdm(range(len(target_image_paths))):
            os.system("cp "+"\"" +target_image_paths[i]+ "\"" + " " +  target_folder)
    ##Get embeddings for clean and noisy images (the lang folder)
    image_folder=lang_folder

    ##Calc embedding start time
    start_time = datetime.now()
    # Get image embeddings
    all_embeddings,image_paths = get_embedding_list(image_folder, timm_pretrained_model=model,
    trained_pth=trained_model_path, device="cuda",
    batch_size=500,transform_type=transform_type,xcit=xcit)
    ##Calc embedding end time
    end_time = datetime.now()

    embedding_time=end_time-start_time
    source_image_paths=[]
    source_image_name=[]
    target_image_paths=[]
    target_image_name=[]
    source_image_embeds=[]
    target_image_embeds=[]
    for i in range(len(image_paths)):
        if "pr19" in image_paths[i] or "concated" in image_paths[i]:
            source_image_paths.append(image_paths[i])
            source_image_embeds.append(all_embeddings[i])
        else:
            target_image_paths.append(image_paths[i])
            target_image_embeds.append(all_embeddings[i])
    
    source_image_embeds=np.array(source_image_embeds)
    target_image_embeds=np.array(target_image_embeds)

    # Get nearest neighbors
    start_time = datetime.now()
    distances, indices = get_nearest_neighbors(source_image_embeds,target_image_embeds, top_k=1)
    end_time = datetime.now()

    nn_time=end_time-start_time
    #Make a dataframe of the results
    bm_df=make_best_match_df(source_image_paths,target_image_paths, indices)

    ##Get output checks through images
    output_check_dir=os.path.join(root_folder,"output_checks")
    if not os.path.exists(output_check_dir):
        os.mkdir(output_check_dir)
    
    ##Write df
    df_path=os.path.join('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm',"bm_df.csv")
    bm_df.to_csv(df_path,index=False)
    ##Get output checks through images
    # ## Make top 10 match dict
    topk_bm_dict=fast_match_topk_dict(source_image_paths,target_image_paths, indices,distances,1)

    def change_target_path(path,root_folder):
        path=os.path.basename(path)
        return os.path.join(root_folder,os.path.basename(path))
    ###Change the paths in the dict - both key and target paths. original dict is of the format {source_path:{targets:[paths]}}
    topk_bm_dict={change_target_path(k,source_root):{"targets":[change_target_path(target_path,target_root) for target_path in  v["targets"]],"distances":v["distances"]} for k,v in topk_bm_dict.items()}

    ##print one example
    print(list(topk_bm_dict.items())[0])

    ###Write dict as json
    dict_path=os.path.join('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm',"top1_bm_dict_all_data.json")
    with open(dict_path, 'w') as fp:
        json.dump(topk_bm_dict, fp)

    ###Conver dict to df. It countains {sources:{targets:[paths], distances:[distances]}}. We need to use the dict
    ###Convert dict to list of tuples (source,target,distance)
    topk_bm_list=[]
    for source,targets_dict in topk_bm_dict.items():
        for target,distance in zip(targets_dict["targets"],targets_dict["distances"]):
            topk_bm_list.append((source,target,distance))
    
    topk_bm_df=pd.DataFrame(topk_bm_list,columns=["source","matched_tk_path","distance"])

    ##Write df
    df_path=os.path.join('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm',"top1_bm_df_all_data_formatted.csv")
    topk_bm_df.to_csv(df_path,index=False)

    print("Embedding time: ",embedding_time)
    print("NN time: ",nn_time)

    ## Calculate match, nomatch accuracy
    print('matched test accuracy:', calculate_matched_accuracy(matched_results = topk_bm_df))
    print('nomatch test accuracy using threshold finetuned on validation set:',calculate_nomatch_accuracy(matched_results = topk_bm_df, file_name="top1_bm_df_all_data_formatted.csv", levenshtein_match = False))

# Run as script
if __name__ == "__main__":
    # Load model
    ##Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit_all_infer_prtkfinal_synthonly', help='path to image folder')
    parser.add_argument('--timm_model', type=str, default="vit_base_patch16_224.dino", help='timm vision transformer model')
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/best_models/enc_best_e_ulti.pth', help='trained model checkpoint')
    parser.add_argument('--lang_code', type=str, default='TK', help='language code - PR is not needed for paper replication')
    parser.add_argument('--transform_type', type=str, default='custom', help='transform type')
    parser.add_argument('--xcit', type=bool, default=False, help='xcit model - deprecated')
    parser.add_argument('--recopy', action="store_true", help='recopy images')

    args = parser.parse_args()
    
    model=args.timm_model
    trained_model_path=args.checkpoint_path
    lang_code=args.lang_code
    
    root_folder = args.root_folder
    recopy = args.recopy

    # Call the main function
    main(root_folder, model, trained_model_path , lang_code, transform_type=args.transform_type,xcit=args.xcit,recopy=recopy)
