###Visualise source and target dataframe and the images contained in them - including target_matched

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read in the dataframes
source_df = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/edit_distance/homo_eff_2_efftk_.csv')

print(len(source_df))
print(source_df.head(4))

###target column contains [''] for all rows. Remove [,', ' and ] from the column
source_df['target'] = source_df['target'].str.replace('[', '')
source_df['target'] = source_df['target'].str.replace(']', '')
source_df['target'] = source_df['target'].str.replace("'", '')
source_df['target'] = source_df['target'].str.replace(" ", '')

print(source_df.head(4))

###Visualise the images. 

#Split in chunks
source_df_chunks=[source_df.filter(items=range(i,i+5),axis=0) for i in range(0,source_df.shape[0],5) ]
        


def visualize_incorrect_matches(incorrect_matches_df_chunk,c_num=0):
    c_num=str(c_num)
##Visualize the incorrect matches using plt and Image. Also print the distance
    fig, ax = plt.subplots(len(incorrect_matches_df_chunk), 3, figsize=(20, 20))
    print(len(incorrect_matches_df_chunk["source"].unique()))
    for i,img_name in enumerate(incorrect_matches_df_chunk["source"].unique()):
        img = Image.open(img_name)
        ax[i,0].imshow(img)
        ax[i,0].axis('off')
        # ax[i,0].set_title("Source")
        img = Image.open(incorrect_matches_df_chunk[incorrect_matches_df_chunk["source"]==img_name]["target"].iloc[0])
        ax[i,1].imshow(img)
        ax[i,1].axis('off')
        # ax[i,1].set_title("Target")
        img = Image.open(incorrect_matches_df_chunk[incorrect_matches_df_chunk["source"]==img_name]["target_matched"].iloc[0])
        ax[i,2].imshow(img)
        ax[i,2].axis('off')


    image_path=f'./homog_incorrect_matches_{c_num}.png'
    ##Save the incorrect matches
    plt.savefig(image_path,dpi=600)

for i,incorrect_matches_chunk in enumerate(source_df_chunks):
    visualize_incorrect_matches(incorrect_matches_chunk,c_num=i)
