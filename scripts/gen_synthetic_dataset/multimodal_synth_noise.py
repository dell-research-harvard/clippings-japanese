
# %%
from numpy.lib.function_base import kaiser
import torchvision.transforms as T
import numpy as np
import os
from tqdm import tqdm
import argparse

from PIL import ImageOps, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm
import numpy as np

from fontTools.unicode import Unicode
import json

#from datasets import color_shift
from create_font_image_folder import load_chars, get_unicode_coverage_from_ttf, get_unicode_chars_font


def pr_color_shift(im):
  choices = np.array([[234,234,212], [225, 207, 171]])
  idx = np.random.choice(range(len(choices)))
  color = [(x + np.random.normal(0, 2))/255 for x in choices[idx]]
  im[0, :, :][im[0, :, :] >= 0.8] = color[0]
  im[1, :, :][im[1, :, :] >= 0.8] = color[1]
  im[2, :, :][im[2, :, :] >= 0.8] = color[2]
  return im


def render_seg(font_paths, unicode_chars, save_path, folder_id, transform, covered_chars_kanji_plus_intersected, font_path_id, random_chars_and_spaces, rand_size): # You can change the imid into folder id
    os.makedirs(os.path.join(save_path,"images",f'{folder_id}'), exist_ok=True)
    os.makedirs(save_path, exist_ok=True) 

    rand_font_path = font_paths[font_path_id] # So we are in one font_path
    digital_font = ImageFont.truetype(rand_font_path, size=rand_size)
    
    coco_bboxes_H, canvas_H = crops_from_text(random_chars_and_spaces, digital_font, font_size=rand_size, 
        n=len(random_chars_and_spaces), horizontal=1)

    coco_bboxes_V, canvas_V = crops_from_text(random_chars_and_spaces, digital_font, font_size=rand_size, 
        n=len(random_chars_and_spaces), horizontal=0)

    font_name = font_paths[font_path_id].split('/')[-1].split('.')[0]
    image_name_H = f"{font_name}_H.png"

    canvas_H_out = transform(canvas_H)
    canvas_H_out.save(os.path.join(save_path, 'images', f'{folder_id}', image_name_H))

    image_name_V = f"{font_name}_V.png"
    
    canvas_V_out = transform(canvas_V)
    canvas_V_out.save(os.path.join(save_path, 'images', f'{folder_id}', image_name_V))

    return coco_bboxes_H, coco_bboxes_V, image_name_H, image_name_V, canvas_H, canvas_V


def crops_from_text(text, font, font_size=256, n=10, horizontal=0):
    # Here you sample the characters and gen the images one by one.
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
        
        coco_bboxes = []
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
            coco_bboxes.append((x, y, w, h))
            x += w
        
        return coco_bboxes, canvas

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
        
        if n==1:
            canvas_h = int(font_size*n+4*font_size)
        else:
            canvas_h = int(font_size*n+4*font_size) # change it larger


        canvas_w = int(font_size*1.3)

        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        
        coco_bboxes = []
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
            coco_bboxes.append((x, y, w, h))
            #x += w # change here
            y += h
        
        return coco_bboxes, canvas
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,required=True, help='save output!')

    args = parser.parse_args()
    synth_transform = T.Compose([
        T.ToTensor(),
        pr_color_shift,
        T.RandomApply([T.GaussianBlur(11)], p=0.35),
        T.ToPILImage()
    ])

    os.makedirs(os.path.join(args.save_path, "coco_files"), exist_ok=True)
    
    # Change: please change the font paths to all available CJK font paths. Put everything in here, and also need to SCP these files to CGIS
    font_paths = ['/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSerifCJKjp-Regular.otf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/HinaMincho-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NewTegomin-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/DelaGothicOne-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/komorebi-gothic.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/ReggaeOne-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/Stick-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/Yomogi-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/HachiMaruPop-Regular.ttf',
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/Kosugi-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/ShipporiMinchoB1-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/Tanugo-TTF-Regular.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSansCJKtc-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/HanaMinB.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/HanaMinA.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSerifCJKtc-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSerifCJKhk-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSansCJKhk-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/SourceHanSansHW-VF.ttf.ttc', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSerifCJKjp-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSansCJKsc-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSerifCJKkr-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSansCJKkr-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSerifCJKsc-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/NotoSansCJKjp-VF.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/taisyokatujippoi7T5.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/sazanami-mincho.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/BabelStoneHan.ttf', 
    '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/Bitstream Cyberbit.ttf']

    char_list = []
    for font_path in tqdm(font_paths):
        font_path=font_path.replace("\\","/")
        char_list.extend(get_unicode_chars_font(font_path))

    char_list=list(set(char_list))
    char_list_int=[str(ord(char)) for char in char_list]

    ###Write integer list to file sepearted by "\n"
    with open(os.path.join(args.save_path,"unicode_chars.txt"), "w") as output:
        output.write("\n".join(char_list_int))

    jis_lvl_1_4_path = os.path.join(args.save_path,"unicode_chars.txt")
    jis_char_info = load_chars(jis_lvl_1_4_path)
    jis_chars = [x[-1] for x in jis_char_info]

    coverage_dict = {}
    train_annotations = []; train_annotations_V = []
    train_images = []; train_images_V = []
    anno_id_from_counter = 0

    for font_path in tqdm(font_paths):
        print(font_path)
        #_, covered_chars = get_unicode_coverage_from_ttf(font_path)
        covered_chars = get_unicode_chars_font(font_path)
        covered_chars_kanji_plus = list(set([c for c in covered_chars]))
        coverage_dict[font_path] = covered_chars_kanji_plus

    covered_chars_kanji_plus_intersected = coverage_dict['/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/yxm/otf/HinaMincho-Regular.ttf']

    char_dict = {}
    # No need to randomly 
    with open('./dataset/longer_no_dup_japanese_word_list.json') as f:
        japanese_word_list = json.load(f)

    for i, random_chars in tqdm(enumerate(japanese_word_list)):
        random_chars_and_spaces = list(random_chars) 
        rand_size = np.random.choice(range(32, 133))
        char_dict[i] = random_chars_and_spaces

        for font_path_id in tqdm(range(0,len(font_paths))):
            signal = 0 # If signal is set to 1 in the for loop, we continue to another font. Skip font part
            for char in random_chars_and_spaces:
                if char not in coverage_dict[font_paths[font_path_id]]:
                    signal = 1

            if signal == 1:
                continue

            coco_bboxes_H, coco_bboxes_V, image_name_H, image_name_V, rimg_H, rimg_V = \
                render_seg(font_paths, jis_chars, args.save_path, i, synth_transform, covered_chars_kanji_plus_intersected, font_path_id, random_chars_and_spaces, rand_size) # The font paths are in, so we can iterate over all fonts. Can directly single out here, do not need the train, test, val
            # Already generated images, these below are for coco json annotations
            imgw_H = int(rimg_H.size[0])
            imgh_H = int(rimg_H.size[1])
            image = {
                "width": imgw_H,
                "height": imgh_H,
                "id": i,
                "file_name": image_name_H
            }
            
            train_images.append(image)

            for bbox in coco_bboxes_H:
                x, y, width, height = bbox
                assert (x >= 0) and (y >= 0)
                if (x + width > imgw_H):
                    width = imgw_H - x - 1
                if (y + height > imgh_H):
                    height = imgh_H - y - 1
                annotation = {
                    "id": anno_id_from_counter, 
                    "image_id": i, 
                    "category_id": 0,
                    "area": int(width*height), 
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "segmentation": [[int(x), int(y), int(x)+int(width), int(y), 
                        int(x)+int(width), int(y)+int(height), int(x), int(y)+int(height)]],
                    "iscrowd": 0,
                    "ignore": 0
                }
                
                train_annotations.append(annotation)

                anno_id_from_counter += 1

        print(len(train_images))


        coco_out_train = {
            "images": train_images,
            "annotations": train_annotations,
            "info": {
                "year": 2021,
                "version": "1.0",
                "contributor": "Synth"
            },
            "categories": [{
                "id": 0, "name": "character"
            }],
            "licenses": ""
        }

        with open(os.path.join(args.save_path, "coco_files", "synth_coco_segs_train_H.json"), 'w') as f:
            json.dump(coco_out_train, f, indent=2)

        # The json output is named as train.json
        # Here we create the vertical json
            imgw_V = int(rimg_V.size[0])
            imgh_V = int(rimg_V.size[1])
            image = {
                "width": imgw_V,
                "height": imgh_V,
                "id": i,
                "file_name": image_name_V
            }
            
            train_images_V.append(image)

            for bbox in coco_bboxes_V:
                x, y, width, height = bbox
                assert (x >= 0) and (y >= 0)
                if (x + width > imgw_V):
                    width = imgw_V - x - 1
                if (y + height > imgh_V):
                    height = imgh_V - y - 1
                annotation = {
                    "id": anno_id_from_counter, 
                    "image_id": i, 
                    "category_id": 0,
                    "area": int(width*height), 
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "segmentation": [[int(x), int(y), int(x)+int(width), int(y), 
                        int(x)+int(width), int(y)+int(height), int(x), int(y)+int(height)]],
                    "iscrowd": 0,
                    "ignore": 0
                }
                
                train_annotations_V.append(annotation)

                anno_id_from_counter += 1

        print(len(train_images))

        # The coco_out_train_V first appears here and do not need to be initialized
        coco_out_train_V = {
            "images": train_images_V,
            "annotations": train_annotations_V,
            "info": {
                "year": 2021,
                "version": "1.0",
                "contributor": "Synth"
            },
            "categories": [{
                "id": 0, "name": "character"
            }],
            "licenses": ""
        }

        with open(os.path.join(args.save_path, "coco_files", "synth_coco_segs_train_V.json"), 'w') as f:
            json.dump(coco_out_train_V, f, indent=2)
