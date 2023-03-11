
# %% Generate words using words from a list (dictionary/place names)
from tkinter import HORIZONTAL, font
from numpy.lib.function_base import kaiser
import torchvision.transforms as T
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from PIL import ImageOps, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm
import json

from fontTools.unicode import Unicode
import json
from glob import glob
from collections import defaultdict
from fontTools.ttLib import TTFont


####Font utils

def load_chars(path):
    with open(path) as f:
        uni = f.read().split("\n")
    return [u.split("\t") for u in uni]

def get_unicode_chars_font(font_file_path):
    """Get all unicode characters in a font file"""
    font = TTFont(font_file_path, fontNumber=0)

    cmap = font['cmap']

 
    cmap_table = cmap.getBestCmap()

    unicode_chars = [chr(c) for c in cmap_table.keys()]
    return unicode_chars


####Render chars
def render_seg(font_paths, save_path, font_path_id, random_chars_and_spaces, rand_size): # You can change the imid into folder id
    # make the folders for this iteration of one font_path_id, you can also iterate over font_path_id inside render_seg

    int_chars=[str(ord(c)) for c in random_chars_and_spaces]
    ###Make the folder for this font_path_id
    folder_name = f"{'_'.join(int_chars)}"
    os.makedirs(os.path.join(save_path,"images",folder_name), exist_ok=True)

    #os.makedirs(os.path.join(save_path, "images"), exist_ok=True)


    # After the chars and words are finalized, we can iterate over the font paths
    #for font_path_id in range(0,len(font_paths)):
    rand_font_path = font_paths[font_path_id] # So we are in one font_path
    digital_font = ImageFont.truetype(rand_font_path, size=rand_size)
    
    # For here, we just generated the text to input, to put text into
    # Since the input for the horizontal and vertical ones are the same, the chars should be the same.
    # For each font
    canvas_H = crops_from_text(random_chars_and_spaces, digital_font, font_size=rand_size, 
        n=len(random_chars_and_spaces), horizontal=1)

    canvas_V = crops_from_text(random_chars_and_spaces, digital_font, font_size=rand_size, 
        n=len(random_chars_and_spaces), horizontal=0)

    #out_image = transform(canvas) # Abhishek mentioned that we don't want any transform here
    font_name = font_paths[font_path_id].split('/')[-1].split('.')[0]
    image_name_H = f"{font_name}-font-{folder_name}-ori-H.png"

    ##Save an image only if it is > 1 pixel in both dim
    if canvas_H.size[0] > 5 and canvas_H.size[1] > 5:


        canvas_H.save(os.path.join(save_path,"images",folder_name,image_name_H))
    

    image_name_V = f"{font_name}-font-{folder_name}-ori-V.png"
    
    if canvas_V.size[0] > 5 and canvas_V.size[1] > 5:

        canvas_V.save(os.path.join(save_path,"images",folder_name,image_name_V))

    return image_name_H,image_name_V,canvas_H, canvas_V


def crops_from_text(text, font, font_size=256, n=10, horizontal=0):
    
    if horizontal == 1:
        p = np.random.uniform(-font_size // 15,font_size // 25)
        crops = []
        
        for c in text:
            img = Image.new('RGB', (font_size*4, font_size*4), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((font_size,font_size), c, (255, 255, 255), font=font, anchor='mm')
            bbox = img.getbbox()
            if bbox is None:
                n -= 1
                continue
            x0,y0,x1,y1 = bbox

            if x1-x0<2*p or y1-y0<2*p:
                pbbox = (x0-font_size // 25,y0-font_size // 25,x1+font_size // 25,y1+font_size // 25)
            else:
                pbbox = (x0-p,y0-p,x1+p,y1+p)
                print('cropped!')
            if int(pbbox[2]-pbbox[0]) == 0 or int(pbbox[3]-pbbox[1]) == 0:
                pbbox = (x0-font_size // 25,y0-font_size // 25,x1+font_size // 25,y1+font_size // 25)

            crop = ImageOps.invert(img.crop(pbbox))
            crops.append(crop)

        
        rand_scal_w = np.random.uniform(1.0, 1.4)
        rand_scal_h = np.random.uniform(1.1, 1.2)
        canvas_w = int(font_size*n*rand_scal_w+font_size*3)
        canvas_h = int(font_size*rand_scal_h+font_size*0.5)
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        
        x = np.random.uniform(0, font_size)
        for i in range(n):
            if text[i] == "_":
                x += font_size
                continue
            if text[i] == ",":
                y = canvas_h - font_size // 3
            else:
                y = font_size // 15
            pcrop = crops[i]
            a = int(pcrop.size[0]*np.random.uniform(0.9, 1.1))+1
            b = int(pcrop.size[1]*np.random.uniform(0.9, 1.1))+1
            print(a)
            print(b)
            pcrop = pcrop.resize((a, 
                b))
            w, h = pcrop.size
            rand_x = np.random.uniform(0, 0.25)
            rand_y = np.random.uniform(0.97, 1.03)
            x = int(w * rand_x) + int(x)
            y = int(y * rand_y)
            # You may also directly crop the pcrop 

            if x + w > canvas_w:
                print(canvas_w-x)
                pcrop = pcrop.resize((int(canvas_w-x),int((canvas_w-x)*h/w)))
                w, h = pcrop.size

            if y + h > canvas_h:
                print(canvas_h-y)
                pcrop = pcrop.resize((int((canvas_h-y)*w/h),int(canvas_h-y)))
                w, h = pcrop.size

            canvas.paste(pcrop, (x, y, x + w, y + h))
            x += w
        
        canvas = ImageOps.invert(canvas)
        bbox = canvas.getbbox()
        if bbox is None:
            pass
        else:
            canvas = canvas.crop(bbox)
        ##Invert again
        canvas = ImageOps.invert(canvas)

        ##Now pad it by 0.3 min size
        canvas = ImageOps.expand(canvas, border=int(0.5*min(canvas.size)), fill='white')

        return canvas

    elif horizontal == 0:
        # Here we want to generate for the vertical versions, here needs to be changed
        p = np.random.uniform(-font_size // 15,font_size // 25) # randomize the p can crop the images
        crops = []

        for c in text:
            img = Image.new('RGB', (font_size*4, font_size*4), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((font_size,font_size), c, (255, 255, 255), font=font, anchor='mm')
            bbox = img.getbbox()
            if bbox is None:
                n -= 1
                continue
            x0,y0,x1,y1 = bbox

            if x1-x0<2*p or y1-y0<2*p:
                pbbox = (x0-font_size // 25,y0-font_size // 25,x1+font_size // 25,y1+font_size // 25)
            else:
                pbbox = (x0-p,y0-p,x1+p,y1+p)
                print('cropped!')
            if int(pbbox[2]-pbbox[0]) == 0 or int(pbbox[3]-pbbox[1]) == 0:
                pbbox = (x0-font_size // 25,y0-font_size // 25,x1+font_size // 25,y1+font_size // 25)

            crop = ImageOps.invert(img.crop(pbbox))
            crops.append(crop)
        
        #rand_scal_w = np.random.uniform(1.0, 1.4)
        #rand_scal_h = np.random.uniform(1.1, 1.2)
        if n==1:
            canvas_h = int(font_size*n+4*font_size)
        else:
            canvas_h = int(font_size*n+4*font_size) # change it larger


        canvas_w = int(font_size*1.3)

        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        
        y = int(0.4*font_size)
        for i in range(n):
            if text[i] == "_":
                y += font_size
                continue
            if text[i] == ",":
                x = canvas_w - font_size // 3
            else:
                x = font_size // 30
            pcrop = crops[i]
            # check the size of width, height
            a = int(pcrop.size[0]*np.random.uniform(0.9, 1.1))+1
            b = int(pcrop.size[1]*np.random.uniform(0.9, 1.1))+1
            print(a)
            print(b)

            pcrop = pcrop.resize((a, 
                b))

            #pcrop = pcrop.resize((int(pcrop.size[0]*np.random.uniform(1, 1.1)), 
            #    int(pcrop.size[1]*np.random.uniform(1, 1.1))))
            # We can apply random crops on these perhaps... crop both width and height
            w, h = pcrop.size


            rand_x = np.random.uniform(0.97, 1.03)
            rand_y = np.random.uniform(0, 0.25)
            x = int(x * rand_x) # We want to change here
            y = int(h * rand_y) + int(y)

            # We don't crop on these
            if x + w > canvas_w:
                print(canvas_w-x)
                pcrop = pcrop.resize((int(canvas_w-x),int((canvas_w-x)*h/w)))
                w, h = pcrop.size

            if y + h > canvas_h:
                print(canvas_h-y)
                pcrop = pcrop.resize((int((canvas_h-y)*w/h),int(canvas_h-y)))
                w, h = pcrop.size

            canvas.paste(pcrop, (x, y, x + w, y + h))
            #x += w # change here
            y += h
        
        ##Now remove all whitespace in the image, get bbox - but do it after inversion
        ##First invert
        canvas = ImageOps.invert(canvas)
        bbox = canvas.getbbox()
        if bbox is None:
            pass
        else:
            canvas = canvas.crop(bbox)
        ##Invert again
        canvas = ImageOps.invert(canvas)

        ##Now pad it by 0.3 min size
        canvas = ImageOps.expand(canvas, border=int(0.3*min(canvas.size)), fill='white')

        return canvas


    

if __name__ == '__main__':

    N = 60000



    # Change the path to CGIS
    
    save_path = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_60000/'
    ##Create folder if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Change: please change the font paths to all available CJK font paths. Put everything in here, and also need to SCP these files to CGIS
    # font_paths = glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/CJK_fonts/*.ttf")    
    font_paths = glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/visual_record_linkage/japan_font_files/*.ttf")
    char_list = []
    for font_path in tqdm(font_paths):
        font_path=font_path.replace("\\","/")
        char_list.extend(get_unicode_chars_font(font_path))

    char_list=list(set(char_list))
    char_list_int=[str(ord(char)) for char in char_list]

    ###Write integer list to file sepearted by "\n"
    with open(save_path+"/char_list.txt", "w") as output:
        output.write("\n".join(char_list_int))

   

    coverage_dict = {}
    ground_truth_dict={}


    for font_path in tqdm(font_paths):
        print(font_path)
        covered_chars = get_unicode_chars_font(font_path)
        covered_chars_kanji_plus = list(set(covered_chars))
        coverage_dict[font_path] = covered_chars_kanji_plus
    with open(save_path+"/coverage_dict.json",'w') as f:
        json.dump(coverage_dict,f,ensure_ascii=False)
    # I think the coverage_dict is enough to get the correct chars. 


    ###Keep only those characters in char list that are covered by at least 3 fonts
    char_list = list(set(char_list))
    char_list = [char for char in char_list if sum([char in coverage_dict[font_path] for font_path in coverage_dict])>=3]

    ##Load place names

    words_covered = []
    path_to_names = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multilang_gts/JP_ja_alt_names.csv"

    ##OPen df, with header in first row
    df = pd.read_csv(path_to_names,header=0)     
    word_list=df['name'].tolist()

    ##Shuffle the word list
    np.random.shuffle(word_list)

    word_list_covered = []
    for i in tqdm(range(0,N)):
        

        word=word_list[i]

        if word in word_list_covered:
            continue

        rand_size = np.random.choice(range(70, 133))
        # The chars are generated

        # We want to pass in the rand_size and random_chars_and_spaces here.

        for font_path_id in tqdm(range(0,len(font_paths))):# generate over different fonts, horizontal/vertical
            # You want to only get these images if the random_chars_and_spaces are all covered

            char_not_covered=False
            for char in word:
                if char not in coverage_dict[font_paths[font_path_id]]:
                    char_not_covered=True
                    break




            try:
                image_name_H, image_name_V, rimg_H, rimg_V = render_seg(font_paths, save_path, font_path_id, word,rand_size) # The font paths are in, so we can iterate over all fonts. Can directly single out here, do not need the train, test, val
                word_list_covered.append(word)

            except:
                print("Error,skipping")
                word_list_covered.append(word)

                continue
