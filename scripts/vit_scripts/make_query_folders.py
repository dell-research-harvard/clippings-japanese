import json
import os
from glob import glob
import random
###Get one image per word in the train and full data


##Load train data anno 
train_anno_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/splits/train.json"
with open(train_anno_path) as f:
    train_anno = json.load(f)


##Generate a folder with 1 image per word - sorted first font, H
folder_list=[dict["file_name"] for dict in train_anno["images"]]

###Clean image root

image_root="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/images" 

save_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/single_font_train"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
##GEnerate a folder with 1 image per word  -  for each folder, sort alphabetically and take only ones containing H.png
for folder in folder_list:
    folder_path = os.path.join(image_root,folder)
    image_list = glob(folder_path+"/*.png")
    image_list = [image for image in image_list if "H.png" in image]
    image_list.sort()
    image = image_list[0]
    image_name = image.split("/")[-1]
    ##Create folder
    if not os.path.exists(os.path.join(save_folder,folder)):
        os.makedirs(os.path.join(save_folder,folder))
    save_path = os.path.join(save_folder,folder,image_name)
    os.system("cp {} {}".format(image,save_path))

###Now similarly, randomly sample 1 image per word for these folders and take only ones containing H.png
save_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/random_font_train"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
for folder in folder_list:
    folder_path = os.path.join(image_root,folder)
    image_list = glob(folder_path+"/*.png")
    image_list = [image for image in image_list if "H.png" in image]
    image_list.sort()
    image = random.choice(image_list)
    image_name = image.split("/")[-1]
    if not os.path.exists(os.path.join(save_folder,folder)):
        os.makedirs(os.path.join(save_folder,folder))
    save_path = os.path.join(save_folder,folder,image_name)
    os.system("cp {} {}".format(image,save_path))





