import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import numpy as np
import os
import math
import json
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from datasets.recognizer_samplers import *
from utils.datasets_utils import *
import re
from tqdm import tqdm


def diff_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class CustomSubset(Dataset):

    def __init__(self, dataset, indices):
        self.super_dataset = dataset
        self.indices = indices
        self.class_to_idx = dataset.class_to_idx
        self.data = [x for idx, x in enumerate(dataset.data) if idx in indices]
        self.targets = [x for idx, x in enumerate(dataset.targets) if idx in indices]

    def __getitem__(self, idx):
        image = self.super_dataset[self.indices[idx]][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.indices)


class WordImageFolder(ImageFolder):

    def __init__(self, root, image_transform=None, patch_resize=False,
                 loader=default_loader, is_valid_file=None):

        super(ImageFolder, self).__init__(root, loader, 
            IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file=is_valid_file)
        self.data = self.samples
        self.image_transform = image_transform
        self.patch_resize = patch_resize

    def __getitem__(self, index):

        path, target = self.data[index]
        sample = self.loader(path)
        
  
        sample = self.image_transform(sample)

        return sample, target


class WordImageFolder_bisect(ImageFolder):
    def __init__(self, root, image_transform=None, patch_resize=False,
                 loader=default_loader, is_valid_file=None,train_stems=None):

        super(ImageFolder, self).__init__(root, loader, 
            IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file=is_valid_file)
        self.data = self.samples
        self.image_transform = image_transform
        self.patch_resize = patch_resize
        self.train_stems = train_stems

        # self.targets = [self.__getitem__(i)[1] for i in range(len(self))]
    def __getitem__(self, index):

        path, target = self.data[index]
        sample = self.loader(path)
        
  
        sample = self.image_transform(sample)
        


        ###if path is in train_stems, then return a tuple target = (target,0) else return a tuple target = (target,1)
        if os.path.basename(path) in self.train_stems:
            target = (target,0)
        else:
            target = (target,1)

        return sample, target



##A class that takes in a dataloader and a transform and returns a new dataloader with the transform applied to each image
class TransformLoader:
    def __init__(self, loader, transform):
        self.loader = loader
        self.transform = transform

    def __iter__(self):
        for data, target in self.loader:
            ##EAch data is a tensor of 252 images. We need to traansform each image 
            ##First, unstack the data by axis 0
            data = torch.unbind(data, dim=0)
            ##Now apply the transform to each image
            data = [self.transform((T.ToPILImage()(word)))  for word in data]
            ##Now stack the images back together
            data = torch.stack(data)

            
            yield data, target

    def __len__(self):
        return len(self.loader)

###Concat 2 datasets together and return a new dataset. Add a dataset identifier to the targets of the combined dataset

def create_dataset(
        root_dir, 
        train_ann_path,
        val_ann_path,
        test_ann_path, 
        batch_size,
        hardmined_txt=None, 
        m=4,
        finetune=False,
        pretrain=False,
        high_blur=False,
        knn=True,
        diff_sizes=False,
        imsize=224,
        num_passes=1,
        resize=True,
        renders=True,
        trans_epoch=False,
        gcd=False,
        unlabeled_dir=None
    ):

    if finetune and pretrain:
        raise NotImplementedError
    if finetune:
        print("Finetuning mode!")
    if pretrain:
        print("Pretraining model!")

    if resize:
        dataset = WordImageFolder(
            root_dir, 
            image_transform= PAD_RESIZE_TRANSFORM if trans_epoch else  create_random_doc_transform()  , #, BASE_TRANSFORM PAD_RESIZE_TRANSFORM ,
            patch_resize=diff_sizes
        )
    else:
        dataset = WordImageFolder(
            root_dir, 
            image_transform= PAD_RESIZE_TRANSFORM if trans_epoch else  create_random_doc_transform_no_resize() , # PAD_RESIZE_TRANSFORM
            patch_resize=diff_sizes
        )

    with open(train_ann_path) as f: 
        train_ann = json.load(f)
        train_stems = [os.path.splitext(x['file_name'])[0] for x in train_ann['images']]
    with open(val_ann_path) as f: 
        val_ann = json.load(f)
        val_stems = [os.path.splitext(x['file_name'])[0] for x in val_ann['images']]
    with open(test_ann_path) as f: 
        test_ann = json.load(f)
        test_stems = [os.path.splitext(x['file_name'])[0] for x in test_ann['images']]

    assert len(set(test_stems).intersection(set(train_stems))) == 0
    assert len(set(val_stems).intersection(set(train_stems))) == 0
    if test_ann_path != val_ann_path:
        assert len(set(val_stems).intersection(set(test_stems))) == 0
    print(f"textline train len: {len(train_stems)}\ntextline val len: {len(val_stems)}\ntextline test len: {len(test_stems)}")
    
    ###
    print("Creating train-test-val splits and checking the splits...")
    if renders:
        paired_train_idx = [idx for idx, (p, t) in tqdm(enumerate(dataset.data)) if \
            any(f'-font-{imf}-ori-' in  os.path.basename(p) for imf in train_stems)]
        paired_val_idx = [idx for idx, (p, t) in tqdm(enumerate(dataset.data)) if \
            any(f'-font-{imf}-ori-' in  os.path.basename(p) for imf in val_stems)]
        paired_test_idx = [idx for idx, (p, t) in tqdm(enumerate(dataset.data)) if \
            any(f'-font-{imf}-ori-' in  os.path.basename(p) for imf in test_stems)]
    else:
        paired_train_idx = [idx for idx, (p, t) in tqdm(enumerate(dataset.data)) if \
            any(imf in  os.path.basename(p) for imf in train_stems)]
        paired_val_idx = [idx for idx, (p, t) in tqdm(enumerate(dataset.data)) if \
            any(imf in  os.path.basename(p) for imf in val_stems)]
        paired_test_idx = [idx for idx, (p, t) in tqdm(enumerate(dataset.data)) if \
            any(imf in  os.path.basename(p) for imf in test_stems)]
    # render_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
    #     not os.path.basename(p).startswith("PAIRED")]
    print(len(set(paired_train_idx).intersection(set(paired_val_idx))))
    assert len(set(paired_train_idx).intersection(set(paired_val_idx))) == 0
    if test_ann_path != val_ann_path:
        assert len(set(paired_val_idx).intersection(set(paired_test_idx))) == 0
    assert len(set(paired_test_idx).intersection(set(paired_train_idx))) == 0 
    print(f"train len: {len(paired_train_idx)}\nval len: {len(paired_val_idx)}\ntest len: {len(paired_test_idx)}")
    
    if finetune:
        idx_train = sorted(paired_train_idx)
    elif pretrain:
        idx_train = sorted(paired_train_idx)
    else:
        idx_train = sorted(paired_train_idx)
    idx_val = sorted(paired_val_idx)
    idx_test = sorted(paired_test_idx)

    if finetune:
        assert len(idx_train) + len(idx_val)  + len(idx_test) == \
            len(dataset), f"{len(idx_train)} + {len(idx_val)} + {len(idx_test)} != {len(dataset)}"
    elif pretrain:
        assert len(idx_train) + len(idx_val)  + len(idx_test) == \
            len(dataset), f"{len(idx_train)} + {len(idx_val)} + {len(idx_test)} != {len(dataset)}"
    else:
        if test_ann_path != val_ann_path:
            assert len(idx_train) + len(idx_val) + len(idx_test) == \
                len(dataset), f"{len(idx_train)} + {len(idx_val)} + {len(idx_test)} != {len(dataset)}"
        else:
            assert len(idx_train) + len(idx_val) == \
                len(dataset), f"{len(idx_train)} + {len(idx_val)} != {len(dataset)}"        

    train_dataset = CustomSubset(dataset, idx_train)
    val_dataset = CustomSubset(dataset, idx_val)
    test_dataset = CustomSubset(dataset, idx_test)
    print(f"Len train dataset: {len(train_dataset)}")
    print(f"Len val dataset: {len(val_dataset)}")
    print(f"Len test dataset: {len(test_dataset)}")

    if hardmined_txt is None:
        train_sampler = NoReplacementMPerClassSampler(
            train_dataset, m=m, batch_size=batch_size, num_passes=num_passes
        )
    else:
        with open(hardmined_txt) as f:
            hard_negatives = f.read().split()
            print(f"Len hard negatives: {len(hard_negatives)}")
        train_sampler = HardNegativeClassSampler(train_dataset, 
            train_dataset.class_to_idx, hard_negatives, m=m, batch_size=batch_size, 
            num_passes=num_passes
        )

    if knn:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True, drop_last=True, 
            sampler=train_sampler, collate_fn=diff_size_collate if diff_sizes else None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True, drop_last=False,
            collate_fn=diff_size_collate if diff_sizes else None)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True, drop_last=False,
            collate_fn=diff_size_collate if diff_sizes else None)





    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=True, 
            collate_fn=diff_size_collate if diff_sizes else None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=False, 
            collate_fn=diff_size_collate if diff_sizes else None)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=False, 
            collate_fn=diff_size_collate if diff_sizes else None)

    if gcd:
        ###Load the unlabelled dataset
        unlabeled_dataset = dataset = WordImageFolder(
            unlabeled_dir, 
            image_transform= PAD_RESIZE_TRANSFORM if trans_epoch else  create_random_doc_transform() , # PAD_RESIZE_TRANSFORM
            patch_resize=diff_sizes
        )

        ###
       
        
        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=True, 
            sampler=None, collate_fn=diff_size_collate if diff_sizes else None)
        
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, unlabeled_dataset, unlabeled_loader
        
    else:

        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader




def create_render_dataset(root_dir):

    image_transform =create_random_doc_transform() # # BASE_TRANSFORM 

    dataset = WordImageFolder(root_dir, image_transform=image_transform)
    idx_render = [idx for idx, (p, t) in enumerate(dataset.data) if "tk_"]

    render_dataset = CustomSubset(dataset, idx_render)
    print(f"Len render dataset: {len(render_dataset)}")
    
    return render_dataset


def create_render_dataset_only_tk(root_dir,tk_path_string="tk"):
    
    image_transform = create_random_doc_transform() #create_random_doc_transform() # #

    dataset = WordImageFolder(root_dir, image_transform=image_transform)
    idx_render = [idx for idx, (p, t) in enumerate(dataset.data) if "tk.png" in p]

    render_dataset = CustomSubset(dataset, idx_render)
    print(f"Len render dataset: {len(render_dataset)}")
    
    return render_dataset
    
def convert_dataset_to_partner_only(dataset,tk_path_string="tk"):
        
        image_transform = create_random_doc_transform() #create_random_doc_transform() # #
    
        idx_render = [idx for idx, (p, t) in enumerate(dataset.data) if "tk.png" not in p and "pr.png" not in p]
    
        subset_dataset = CustomSubset(dataset, idx_render)
        print(f"Len render dataset: {len(subset_dataset)}")
        
        return subset_dataset