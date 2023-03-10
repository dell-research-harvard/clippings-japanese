###Models
import numpy as np
import pandas as pd
import timm
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision.transforms as T

import faiss
import math

from transformers import AutoModel,AutoTokenizer, AutoModelForMaskedLM

import torch
import timm
from timm.data import resolve_data_config, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T

from sentence_transformers import SentenceTransformer 



###MLP
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




###Encoder - takes in image and text, feeds it to a vision model (like VIT) and a text model (like SBERT), concatenates and then feeds the output to an MLP
class MultiModalEncoder(nn.Module):
    def __init__(self, vision_model, text_model, output_dim=768, hidden_dim=2048, num_layers=1, dropout=0.1):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = MLP(2 * 768, self.output_dim, self.hidden_dim, self.num_layers, self.dropout)
        self.text_model.to("cuda")
        self.vision_model.to("cuda")
 

    def forward(self, image, text, freeze_vision=True, freeze_text=True):
        # print("\tIn Model: image size", image.size())
        # ###text is a dict . Print sizess
        # print("\tIn Model: input_ids size", text['input_ids'].size())
        # print("\tIn Model: attention_mask size", text['attention_mask'].size())
        # print("\tIn Model: token_type_ids size", text['token_type_ids'].size())
        if freeze_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            with torch.no_grad():
                image = self.vision_model(image)
                
        else:
            image = self.vision_model(image)
            ##Normalise
        image = F.normalize(image, p=2, dim=1)
        if freeze_text:
            ###check if layers are frozen
            self.text_model=self.text_model.set_parameter_requires_grad(False)

            with torch.no_grad():
                text = self.text_model.forward(text)
        else:
            text = self.text_model.forward(text)
        ###Normalise
        text = F.normalize(text, p=2, dim=1)
        x = torch.cat([image, text], dim=1)
        ###Normalise the concatenated embeddings
        x = F.normalize(x, p=2, dim=1)
        return self.model(x)


class MultiModalPoolingEncoder(nn.Module):
    def __init__(self, vision_model, text_model,pooling_type="mean"):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.pooling_type= pooling_type

    def forward(self, image, text, freeze_vision=False, freeze_text=False):
        if freeze_vision:
            with torch.no_grad():
                image = self.vision_model(image)
        else:
            image = self.vision_model(image)
        image=F.normalize(image, p=2, dim=1)
        if freeze_text:
            with torch.no_grad():
                text = self.text_model.forward(text)
        else:
            text = self.text_model.forward(text)
        text=F.normalize(text, p=2, dim=1)
        ###Pool the image and text embeddings - take the mean of the embeddings combined
        if self.pooling_type=="mean":
            x = (torch.add(image,text))*0.5
        elif self.pooling_type=="max":
            x = torch.max(image,text)
        elif self.pooling_type=="concat":
            x = torch.cat([image, text], dim=1)
        x=F.normalize(x, p=2, dim=1)
        return x

###Add positional encodings to the embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

###A multimodal encoder that has a transformer instead of an MLP
class MultiModalTransformerEncoder(nn.Module):
    def __init__(self, vision_model, text_model, nhead=8, num_encoder_layers=12, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.dropout = dropout
        # self.model = nn.Transformer(d_model=1536, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768,nhead=8), num_layers=num_encoder_layers)
    def forward(self, image, text, freeze_vision=True, freeze_text=True):
        if freeze_vision:
            with torch.no_grad():
                image = self.vision_model.forward_features(image)
        else:
            image = self.vision_model.forward_features(image)
        if freeze_text:
            with torch.no_grad():
                text = self.text_model.forward_cls_token_embeddings(text)
        else:
            text = self.text_model.forward_cls_token_embeddings(text)
        #

        ##note the position of the cls token in image and text embeddings is 0.
        x = torch.cat([image, text], dim=1)
        x = F.normalize(x, p=2, dim=1)
        transformer_output=self.model(x)
        image_cls_token = transformer_output[:,0,:]
        text_cls_token = transformer_output[:,image.shape[1],:]

        return image_cls_token, text_cls_token
    




def AutoVisionEncoderFactory(backend, modelpath):

    if backend == "timm":

        class AutoEncoder(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = timm.create_model(model, num_classes=0, pretrained=True,img_size=224)
                net.to(device)
                self.net = net

            def forward(self, x):
                x = self.net(x)
                return x 
            
            def forward_features(self,x):
                x = self.net.forward_features(x)[:,0:]
                # # print("Shape of vision features")
                # print(x.shape)
                return x

            


            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    elif backend == "hf":

        class AutoEncoder(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = AutoModel.from_pretrained(model)
                net.to(device)
                self.net = net

            def forward(self, x):
                x = self.net(x)
                x = x.last_hidden_state[:,0,:]
                return x

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    else:
        
        raise NotImplementedError

    return AutoEncoder


###Sbert encoder
class SBERTEncoder:
    def __init__(self, model_path):
        super().__init__()
        self.model = SentenceTransformer(model_path)
        self.output_dim = self.model.get_sentence_embedding_dimension()

    
    def forward(self, tokenised_sentences):
        # print("-----------------------------------------------------")
        # print("\tIn Model: input_ids size", tokenised_sentences['input_ids'].size())
        # print("\tIn Model: attention_mask size", tokenised_sentences['attention_mask'].size())
        # print("\tIn Model: token_type_ids size", tokenised_sentences['token_type_ids'].size())
        return self.model.forward(tokenised_sentences)["sentence_embedding"]
    
    def encode(self, sentences):
        return self.model.encode(sentences)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def set_parameter_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad
        return self

    def check_if_layers_frozen(self):
        for param in self.model.parameters():
            print(param.requires_grad)
        return self
    
    def forward_token_embeddings(self, tokenised_sentences):
        x= self.model.forward(tokenised_sentences)["token_embeddings"]
        print("Shape of text features")
        print(x.shape)
        return x
    
    def forward_cls_token_embeddings(self, tokenised_sentences):
        return self.model.forward(tokenised_sentences)["cls_token_embeddings"]
    
    def forward_cls_token_embeddings(self, tokenised_sentences):
        '''Add cls token to the token embeddings'''
        cls_token=self.model.forward(tokenised_sentences)["cls_token_embeddings"]
        token_embeddings=self.model.forward(tokenised_sentences)["token_embeddings"]
        ###Add the cls token to the token embeddings
        x=torch.cat([cls_token.unsqueeze(1),token_embeddings],dim=1)
        return x
    


###Huggingface bert encoder
class HFEncoder:
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.output_dim = self.model.config.hidden_size

    def forward(self, tokenised_sentences):
        x = self.model.forward(**tokenised_sentences)[0][:,0,:]
        print("Shape of text features",x.shape)
        return self.model.forward(**tokenised_sentences)[0][:,0,:]
    
    def encode(self, sentences):
        return self.model.encode(sentences)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def set_parameter_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad
        return self

    def check_if_layers_frozen(self):
        for param in self.model.parameters():
            print(param.requires_grad)
        return self
    
    def forward_token_embeddings(self, tokenised_sentences):
        x= self.model.forward(**tokenised_sentences)[0]
        print("Shape of text features",x.shape)        
        return self.model.forward(**tokenised_sentences)[0]
    
    def forward_cls_token_embeddings(self, tokenised_sentences):
        x= self.model.forward(**tokenised_sentences)[0][:,0,:]
        print("Shape of text features",x.shape)
        return self.model.forward(**tokenised_sentences)[0][:,0,:]
    
    def forward_cls_token_embeddings(self, tokenised_sentences):
        '''Add cls token to the token embeddings'''
        cls_token=self.model.forward(**tokenised_sentences)[0][:,0,:]
        token_embeddings=self.model.forward(**tokenised_sentences)[0]
        ###Add the cls token to the token embeddings
        x=torch.cat([cls_token.unsqueeze(1),token_embeddings],dim=1)
        print("Shape of text features",x.shape)
        return x
    
class BertEncoder:

    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.output_dim = self.model.config.hidden_size
        self.tokenizer= AutoTokenizer.from_pretrained(model_path)

    def forward(self, tokenised_sentences):
        return self.model.forward(**tokenised_sentences)[0][:,0,:]
    
    def encode(self, sentences):
        return self.model.encode(sentences)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def set_parameter_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad
        return self

    def check_if_layers_frozen(self):
        for param in self.model.parameters():
            print(param.requires_grad)
        return self
    
    def forward_token_embeddings(self, tokenised_sentences):
        return self.model.forward(**tokenised_sentences)[0]
    
    def forward_cls_token_embeddings(self, tokenised_sentences):
        return self.model.forward(**tokenised_sentences)[0][:,0,:]
    
    def forward_cls_token_embeddings(self, tokenised_sentences):
        '''Add cls token to the token embeddings'''
        cls_token=self.model.forward(**tokenised_sentences)[0][:,0,:]
        token_embeddings=self.model.forward(**tokenised_sentences)[0]
        ###Add the cls token to the token embeddings
        x=torch.cat([cls_token.unsqueeze(1),token_embeddings],dim=1)
        return x
    def tokenize(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
