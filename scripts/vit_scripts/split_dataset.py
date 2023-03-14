####Create train-test-val splits - distribute folders

import json
from glob import glob
import random
import os


####Run as script

if __name__ == "__main__":

    data_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/images"

    dataset_paths=glob(data_dir+"/*")

    split_prop = [0.6,0.2,0.2]

    train_paths = []
    test_paths = []
    val_paths = []

    ##SShuffle dataset paths
    random.shuffle(dataset_paths)


    for i in range(len(dataset_paths)):
        path = dataset_paths[i]
        if i < len(dataset_paths)*split_prop[0]:
            train_paths.append(path)
        elif i < len(dataset_paths)*(split_prop[0]+split_prop[1]):
            test_paths.append(path)
        else:
            val_paths.append(path)

    ##Save paths to json {images:[{file_name:file_i}] }

    train_list = []
    test_list = []
    val_list = []

    for i in range(len(train_paths)):
        path = train_paths[i]
        train_list.append({"file_name":path.split("/")[-1]})
    
    train_dict = {"images":train_list}

    for i in range(len(test_paths)):
        path = test_paths[i]
        test_list.append({"file_name":path.split("/")[-1]})
    
    test_dict = {"images":test_list}
    
    for i in range(len(val_paths)):
        path = val_paths[i]
        val_list.append({"file_name":path.split("/")[-1]})

    val_dict = {"images":val_list}


    
    ###SAve to json


    save_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/splits"

    ###Make the folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    with open(os.path.join(save_path,"train.json"),"w") as f:
        json.dump(train_dict,f)

    with open(os.path.join(save_path,"test.json"),"w") as f:
        json.dump(test_dict,f)

    with open(os.path.join(save_path,"val.json"),"w") as f:
        json.dump(val_dict,f)

    print(train_dict.keys())
    


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




        