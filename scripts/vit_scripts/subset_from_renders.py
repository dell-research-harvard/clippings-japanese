##Subset data - copy only some folders

import glob
import os

data_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_60000/images"

dataset_paths=glob.glob(data_dir+"/*")

subset_paths = dataset_paths[:40000]
print(len(subset_paths))


output_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_40000/"

##MAke the output dir
os.mkdir(output_dir)

##Make an image dir
os.mkdir(output_dir+"images")

##Copy the images
for path in subset_paths:
    os.system("cp "+path+" "+output_dir+"images/" + " -r")

