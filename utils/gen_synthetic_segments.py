import os
from PIL import  Image
import numpy as np
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
import random
from tqdm import tqdm
from glob import glob
import torch





####Some functions and transforms for augmenting images
####The augraphy package is also supported - but never used for training


#####Image transform functions
def blur_transform(high):
    if high:
        return T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.3)
    else:
        return  T.RandomApply([T.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.3)



LIGHT_AUG =  T.Compose([
        ##TO RGB
        T.ToPILImage(),
        T.Lambda(lambda x: x.convert("RGB") if not torch.is_tensor(x) else x),
        T.ToTensor(),
        # T.RandomApply([T.RandomAffine(degrees=2, translate=(0.1,0.1), fill=1)], p=0.7),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)], p=0.5),
        T.ToPILImage(),
        T.RandomGrayscale(p=0.2),
        T.RandomInvert(p=0.05),
        # T.RandomAutocontrast(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        
    ])


def create_render_transform(high_blur,affine_degrees,translate,scale,contrast,brightness,saturation,hue,p_noise):
    """ This has language based operations - harmonise later"""
    return T.Compose([
        ##TO RGB
        T.Lambda(lambda x: x.convert("RGB") if not torch.is_tensor(x) else x),
        T.ToTensor(),
        T.RandomApply([T.RandomAffine(degrees=affine_degrees, translate=translate, fill=1)], p=0.7),
        T.RandomApply([T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)], p=0.5),
        T.ToPILImage(),
        lambda x: Image.fromarray(A.GaussNoise(var_limit=(10.0, 40.0), mean=0, p=0.5)(image=np.array(x))["image"]),
        blur_transform(high_blur),
        T.RandomGrayscale(p=0.2),
        T.RandomInvert(p=0.05),
        T.RandomPosterize(2, p=0.05),
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAutocontrast(), ##dISABLE FOR FINETUNING
        T.RandomEqualize(), ##dISABLE FOR FINETUNING
        T.ToTensor(),
        T.ToPILImage()
    ])



###Tuples - scale,translate, all params are between 0,1 , translate has to have second number greater than 1st
###Randomly draw params and save them to a dict
def get_render_transform_params():
    scale = (1,1)
      ##Two copies of the same number randomly drawn
    translate_num = (random.uniform(0,0.1))
    translate = (translate_num,translate_num)
    affine_degrees = 2 # random.uniform(0, 2)
    contrast = 0.2 #random.uniform(0.2, 0.4)
    brightness = 0.4 # random.uniform(0.4, 0.6)
    saturation = 0.2 # random.uniform(0.2, 0.4)
    hue = 0.2 # random.uniform(0.1, 0.4)
    p_noise =  0.1 # random.uniform(0.1, 0.2)
    high_blur = random.choice([False,True])
    return {"high_blur": high_blur,"affine_degrees": affine_degrees,"scale": scale, "translate": translate, "contrast": contrast,
            "brightness": brightness, "saturation": saturation, "hue": hue, "p_noise": p_noise}


def get_render_transform():
    params = get_render_transform_params()
    return create_render_transform(**params)


def create_synthetic_images(im_subfolder_path,save_dir,image_count):
    image_path_list = glob(im_subfolder_path+"/*.png")
    for i in range(len(image_path_list)):
        img_path = image_path_list[i]
        img_name= img_path.split("/")[-1]
        img = Image.open(img_path)
        img = img.convert("RGB")
        
        subfolder_name = im_subfolder_path.split("/")[-1]
        folder_path = os.path.join(save_dir,str(subfolder_name))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for j in tqdm(range(image_count)):
            transforms = get_render_transform()
            img = transforms(img)
            img.save(os.path.join(folder_path,"transform_" + str(j)+img_name))



##Run as script (test)
if __name__ == "__main__":

    clean_images_dir= "/path/to/data/word_dump_cjk/images"
    noisy_dir = "/path/to/data/word_dump_cjk/noisy_images"
    image_count = 20

    ##Create save dir if it doesn't exist
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)
    
    ##Subfolder list
    subfolder_list = glob(clean_images_dir+"/*")
    print(subfolder_list)

    ##Create synthetic images for each subfolder in parallel, add a progress bar
    # with Pool(64) as p:
    #     list=p.imap_unordered(partial(create_synthetic_images,noisy_dir,20),subfolder_list)

    create_synthetic_images(subfolder_list[0],noisy_dir,20)
        


        

