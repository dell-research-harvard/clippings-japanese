import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from torchvision import transforms as T
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
import kornia
from albumentations.pytorch import ToTensorV2
import utils.gen_synthetic_segments as gss
import glob 



GRAY_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=3),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


INV_NORMALIZE = T.Normalize(
   mean= [-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
   std= [1/s for s in IMAGENET_DEFAULT_STD]
)






class MedianPad:

    def __init__(self, override=None):

        self.override = override

    def __call__(self, image):

        ##Convert to RGB 
        image = image.convert("RGB") if isinstance(image, Image.Image) else image
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        pad_x, pad_y = [max_side - s for s in image.size]
        # padding = (0, 0, pad_x, pad_y)
        padding = (round((10+pad_x)/2), round((5+pad_y)/2), round((10+pad_x)/2), round((5+pad_y)/2)) ##Added some extra to avoid info on the long edge

        imgarray = np.array(image)
        h, w , c= imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        return T.Pad(padding, fill=medval if self.override is None else self.override)(image)






BASE_TRANSFORM = T.Compose([
        MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

NO_PAD_NO_RESIZE_TRANSFORM = T.Compose([
    ##To rgb
        T.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
        T.ToTensor(),
        T.Resize(224,max_size=225),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

LIGHT_AUG_BASE= T.Compose([T.Normalize(
   mean= [-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
   std= [1/s for s in IMAGENET_DEFAULT_STD]),gss.LIGHT_AUG, BASE_TRANSFORM])

EMPTY_TRANSFORM = T.Compose([
        T.ToTensor()
])


PAD_RESIZE_TRANSFORM = T.Compose([
        MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224))
])


AUG_NORMALIZE = T.Compose([gss.random_image_transform(), T.ToTensor(),
 T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])


RANDOM_TRANS_PAD_RESIZE= T.Compose([gss.random_image_transform(), PAD_RESIZE_TRANSFORM])

def create_random_doc_transform(size=224):
    return T.Compose([  gss.random_image_transform(), BASE_TRANSFORM])


def create_clip_random_doc_transform(size=224):
    return T.Compose([  gss.random_image_transform(), CLIP_BASE_TRANSFORM])


def create_random_no_aug_doc_transform(size=224):
    return T.Compose([  gss.get_render_transform(), BASE_TRANSFORM])
    
def create_random_doc_transform_no_resize():
    return T.Compose([  gss.random_image_transform(), NO_PAD_NO_RESIZE_TRANSFORM])

CLIP_BASE_TRANSFORM = T.Compose([  
         MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])
##Run as script
if __name__ == "__main__":


    ##Test transform on image
    # img_path = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan/images/67_24161_25940_32160_22808_19991_36893_21890_31105_34597_35595_27137_33531_31796_31013_26057/komorebi-gothic-font-67_24161_25940_32160_22808_19991_36893_21890_31105_34597_35595_27137_33531_31796_31013_26057-ori-H.png"
    # img = Image.open(img_path)

    # img_base = BASE_TRANSFORM(img)
    # img_no_pad_base= NO_PAD_BASE_TRANSFORM(img)
    # img_not_norm = gss.random_image_transform()(img)
    # img_random = create_random_doc_transform()(img)
    
    # ##Convert tensor to image
    # img_base = T.ToPILImage()(img_base)
    # img_random = T.ToPILImage()(img_random)
    # img_no_pad_base = T.ToPILImage()(img_no_pad_base)
    # # img_not_norm = T.ToPILImage()(img_not_norm)
    # ##save images
    # img_base.save("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan/base_1.png")
    # img_random.save("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan/random_1.png")
    # img_not_norm.save("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan/not_norm_1.png")
    # img_no_pad_base.save("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan/no_pad_base_1.png")

    ###Test transforms on all images to find error
    save_dir= "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_japan_centered_3000/trans_images"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter=0

    img_list = glob.glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_japan_centered_3000/images/*/*.png")
    for i, img_path in enumerate(img_list):
        ##open image
        print(i, img_path)
        img = Image.open(img_path)
        ##Transform image

        # img_base = BASE_TRANSFORM(img)
        print("random transform")
        img_random= create_random_doc_transform()(img)
        ##Save
        img_random = T.ToPILImage()(img_random)
        img_random.save(os.path.join(save_dir, os.path.basename(img_path)))

        counter+=1
        if counter>100:
            break
    

