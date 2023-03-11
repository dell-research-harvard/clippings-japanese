from transformers import (
    ViTModel, BeitModel, 
    SwinModel, ConvNextModel,
    ViTMAEModel, AutoModel
)
import torch
import timm
from timm.data import resolve_data_config, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T
import torch.nn as nn


class XcitEncoder(torch.nn.Module):

    def __init__(self, 
            timm_model='xcit_small_12_p8_224',
            fb_model='https://dl.fbaipublicfiles.com/xcit/xcit_small_12_cp8_dino.pth',
            device='cuda'
        ):
        super().__init__()
        net = timm.create_model(timm_model, num_classes=0, pretrained=False)
        net.to(device)
        checkpoint = torch.hub.load_state_dict_from_url(fb_model, map_location=device, check_hash=True)
        checkpoint = timm.models.xcit.checkpoint_filter_fn(checkpoint, net)
        net.load_state_dict(checkpoint, strict=True)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x

    @classmethod
    def load(cls, checkpoint):
        ptnet = cls()
        ptnet.load_state_dict(torch.load(checkpoint))
        return ptnet


class BitEncoder(torch.nn.Module):

    def __init__(self, 
            timm_model='resnetv2_101x1_bitm_in21k',
            device='cuda'
        ):
        super().__init__()
        net = timm.create_model(timm_model, num_classes=0, pretrained=True)
        net.to(device)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x 

    @classmethod
    def load(cls, checkpoint):
        ptnet = cls()
        ptnet.load_state_dict(torch.load(checkpoint))
        return ptnet


class MaeEncoder(torch.nn.Module):

    def __init__(self, hub_url="facebook/vit-mae-base", device='cuda'):
        super().__init__()
        net = ViTMAEModel.from_pretrained(hub_url)
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


class VitEncoder(torch.nn.Module):

    def __init__(self, hub_url='facebook/dino-vitb16', device='cuda'):
        super().__init__()
        net = ViTModel.from_pretrained(hub_url)
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


class BeitEncoder(torch.nn.Module):

    def __init__(self, hub_url='microsoft/beit-large-patch16-384', device='cuda'):
        super().__init__()
        net = BeitModel.from_pretrained(hub_url)
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


class SwinEncoder(torch.nn.Module):

    def __init__(self, hub_url='microsoft/swin-base-patch4-window7-224-in22k', device='cuda'):
        super().__init__()
        net = SwinModel.from_pretrained(hub_url)
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


class ConvNextEncoder(torch.nn.Module):

    def __init__(self, hub_url='facebook/convnext-base-224-22k', device='cuda'):
        super().__init__()
        net = ConvNextModel.from_pretrained(hub_url)
        net.to(device)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        x = x.pooler_output
        return x

    @classmethod
    def load(cls, checkpoint):
        ptnet = cls()
        ptnet.load_state_dict(torch.load(checkpoint))
        return ptnet


class ProjectionHead(torch.nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        return self.model(x)


def AutoEncoderFactory(backend, modelpath):

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

            # def forward(self, x):
            #     x = self.net.forward_features(x)[:,1:]
            #     ###Try: Take weighted average of all the patches, deeper layers have more weight
            #     x = torch.mean(x, dim=1)
            #     return x

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