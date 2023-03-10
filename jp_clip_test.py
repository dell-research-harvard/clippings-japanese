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


import matplotlib.pyplot as plt
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

def get_image_text_embeddings(data_loader,model,device):
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
                ###MEan of the two embeddings
                final_embeds=(image_embeds*0.4 +  text_embeds*0.6)
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

if __name__ == "__main__":
    data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_train.csv"
    data=pd.read_csv(data_path)
    print("original size of data: {}".format(len(data)))

    all_text=data["text"].unique()

    ###drop duplicates if image path and text are the same
    data=data.drop_duplicates(subset=['image_path','text'])


    device = "cuda" if torch.cuda.is_available() else "cpu"


    model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
    cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/clip_pretrain_unlabelled_m1_v2.pt"
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/imwt_4bienc_clip_pretrain_labelled_m3_v2.pt"
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/imwt_5bienc_clip_pretrain_labelled_m3_v3.pt"
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v2_hardneg_synth_cp.pt"
    cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_hardneg_aug_cp_synth_hardneg.pt"
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/clip_pretrain_unlabelled_m1_v3.pt"
   
   
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_39clip_pretrain_unlabelled_m1_v3.pt"
    # # model.load_state_dict(clean_checkpoint(cp_path, use_ema=True, clean_aux_bn=False,clean_net=True))
    # ##Loadthe checkpoint
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt"

    model.load_state_dict((clean_checkpoint(cp_path)))
        

    model.to(device)
    tokenizer = ja_clip.load_tokenizer()

    img = Image.open("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/tk_title_img_1025_v2/tk1957_0028_2_14_title_20339.png")
    image = CLIP_BASE_TRANSFORM(img).unsqueeze(0).to(device)
    encodings = ja_clip.tokenize(
        texts=["会社名(小川厚一商店)は次のように書かれています", "会社名(丸善石油)は次のように書かれています","会社名(民生デイゼル工業)は次のように書かれています"],
        max_seq_len=77,
        device=device,
        tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
    )

    with torch.no_grad():
        image_features = model.get_image_features(image)
        ##Normalise
        image_features=image_features/torch.norm(image_features, dim=1, keepdim=True)
        text_features = model.get_text_features(**encodings)
        text_features=text_features/torch.norm(text_features, dim=1, keepdim=True)

        print("Image features:", image_features.shape)

        print("Text features:", text_features.shape)
        
        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        ###Check cosine similarity between image and text
        for i in range(len(text_features)):
            print(text_features[i].shape)
            print("Cosine similarity between image and text: {}".format(cosine_similarity_torch(image_features[0],text_features[i])))
         
        
        ##Load ground truth
        ocr_tk_match_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk/TK_matched_ocr.csv"

        ocr_tk_match_df=pd.read_csv(ocr_tk_match_path)

        ##GEt only targets realted variables
        ocr_tk_match_df=ocr_tk_match_df[['target','target_text_gt']]

        ##Rename columns
        ocr_tk_match_df=ocr_tk_match_df.rename(columns={'target':'image_path','target_text_gt':'text'})





        ##Load test data 
        tk_universe=get_tk_universe()
        test_data_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_test.csv"
        test_data = pd.read_csv(test_data_path)
        ###Drop duplicates by image_path and text
        test_data=test_data.drop_duplicates(subset=['image_path','text'])
        

        print(test_data.head(5))

        ##Give more labguage to the 'text'. This is a hack to make the model learn better
        # test_data['text']=test_data['text'].apply(lambda x: "会社名(" + x + ")は次のように書かれています")
        # tk_universe['text']=tk_universe['text'].apply(lambda x: "会社名(" + x + ")は次のように書かれています")

        ##SAmple 1000 from tk_universe
        # tk_universe=tk_universe.sample(n=1000, random_state=42)

        ###Add ocr_tk_match_path to tk_universe
        tk_universe=tk_universe.append(ocr_tk_match_df, ignore_index=True)

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

        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=300, shuffle=False, num_workers=16)
        tk_universe_loader = torch.utils.data.DataLoader(tk_universe, batch_size=300, shuffle=False, num_workers=16)


        test_embeddings, test_labels, test_text, test_paths = get_image_text_embeddings(test_data_loader,model, device)
        tk_universe_embeddings, tk_universe_labels, tk_universe_text, tk_paths = get_image_text_embeddings(tk_universe_loader,model, device)


        print("test embeddings shape: {}".format(test_embeddings.shape))

        ###Make an index using tk universe
        index = faiss.IndexFlatIP(test_embeddings.shape[1])
        index.add(tk_universe_embeddings.cpu().numpy())

        ###Get top 1 match
        D, I = index.search(test_embeddings.cpu().numpy(), 1)

        ###Print corresponding text of test and tk universe match

        ###Make a dataframe of test image path and tk universe match path
        test_tk_match=pd.DataFrame()
        test_tk_match["source"]=test_paths
        test_tk_match["matched_tk_path"]=[tk_paths[i[0]] for i in I]
        test_tk_match["test_text"]=test_text
        test_tk_match["matched_tk_text"]=[tk_universe_text[i[0]] for i in I]
        ###GEt distances as well
        test_tk_match["distance"]=[i[0] for i in D]



        ###Now merge with test data
        ground_truth_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_pr_tk/TK_matched.csv"

        ground_truth=pd.read_csv(ground_truth_path)

        ###Now merge with ground truth on source
        test_tk_match=pd.merge(test_tk_match,ground_truth,on="source",how="left")

        ##Now calculate accuracy if matched_tk_path is same as target
        test_tk_match["accuracy"]=test_tk_match.apply(lambda x: 1 if x["matched_tk_path"]==x["target"] else 0,axis=1)

        print(test_tk_match.head(5))

        print("Accuracy: {}".format(test_tk_match["accuracy"].sum()/len(test_tk_match)))

        incorrect_match_df=test_tk_match[test_tk_match["accuracy"]==0]

        ###
        # incorrect_matches_chunks=[incorrect_match_df[i:i+5] for i in range(0,incorrect_match_df.shape[0],5)]
        ##Reset index
        incorrect_match_df=incorrect_match_df.reset_index(drop=True)

        # incorrect_matches_chunks=[incorrect_match_df.filter(items=range(i,i+5),axis=0) for i in range(0,incorrect_match_df.shape[0],5) ]
        


        def visualize_incorrect_matches(incorrect_matches_df,c_num=0):
            c_num=str(c_num)
        ##Visualize the incorrect matches using plt and Image. Also print the distance
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
                img = Image.open(incorrect_matches_df[incorrect_matches_df["source"]==img_name]["matched_tk_path"].iloc[0])
                ax[i,2].imshow(img)
                ax[i,2].axis('off')
                ###Also print the distance
                ax[i,2].set_title("Distance: {}".format(incorrect_matches_df[incorrect_matches_df["source"]==img_name]["distance"].iloc[0]))


            image_path=f'./incorrect_matches_{c_num}.png'
            ##Save the incorrect matches
            plt.savefig(image_path,dpi=600)

        # for i,incorrect_matches_chunk in enumerate(incorrect_matches_chunks):
        #     visualize_incorrect_matches(incorrect_matches_chunk,c_num=i)

        visualize_incorrect_matches(incorrect_match_df,c_num=0)

        


        




