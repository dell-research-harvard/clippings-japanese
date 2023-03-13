import glob
import PIL
from PIL import ImageOps, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm
import numpy as np
from torchvision import transforms as T
from torch import nn
import argparse

from fontTools.ttLib import TTFont
from itertools import chain
from fontTools.unicode import Unicode
from collections import defaultdict


def ext_fname(x):
    return os.path.splitext(os.path.basename(x))[0]


def load_chars(path):
    with open(path) as f:
        uni = f.read().split("\n")
    return [u.split("\t") for u in uni]


def draw_single_char(ch, font, canvas_size, padding=0.):
    img = Image.new("L", (canvas_size * 4, canvas_size * 4), 0)
    c_w, c_h = img.size
    draw = ImageDraw.Draw(img)
    try:
        draw.text(
            (c_w // 2, c_h // 2), 
            ch, canvas_size, font=font, 
            anchor="mm"
        )
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    # l, u = max(0, l - 5), max(0, u - 5)
    # r, d = min(canvas_size * 2 - 1, r + 5), min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    xdist, ydist = abs(l-r), abs(u-d)
    img = np.array(img)
    img = img[u-int(padding*ydist):d+int(padding*ydist), l-int(padding*xdist):r+int(padding*xdist)]
    img = 255 - img
    img = Image.fromarray(img)
    width, height = img.size
    try:
        img = T.ToTensor()(img)
    except SystemError as e:
        print(e)
        return None
    img = img.unsqueeze(0) 
    pad_len = int(abs(width - height) / 2)  
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    img = img.squeeze(0)
    img = T.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


def draw_single_char_ascender(ch, font, canvas_size, padding=0.):
    canvas_width, canvas_height = (canvas_size * 5, canvas_size * 5)
    img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0,0), ch, (255, 255, 255), font=font)
    bbox = img.getbbox()
    w, h = font.getsize(ch)
    x0, y0, x1, y1 = bbox
    vdist, hdist = y1-y0, x1-x0
    x0, y0, x1, h = x0-(hdist*padding), y0-(vdist*padding), x1+(hdist*padding), h+(vdist*padding)
    uninverted_image = img.crop((x0, 0, x1, h))
    return ImageOps.invert(uninverted_image)


def get_unicode_coverage_from_ttf(ttf_path):
    with TTFont(ttf_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1) as ttf:
        chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
        chars_dec = [x[0] for x in chars]
        return chars_dec, [chr(x) for x in chars_dec]


def filter_recurring_hash(charset, font, canvas_size):
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size)
        if img is not None:
            hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def render_chars(font_paths, unicode_chars, save_path, padding=0., draw_func=draw_single_char, nonsquare=False):

    os.makedirs(save_path, exist_ok=True)
    idx = 0

    for font_path in font_paths:

        print(font_path)
        font_name, _ = os.path.splitext(os.path.basename(font_path))

        digital_font = ImageFont.truetype(font_path, size=256)
        _, covered_chars = get_unicode_coverage_from_ttf(font_path)
        covered_chars_kanji_plus = list(set([c for c in covered_chars if c in unicode_chars]))

        filter_hashes = set(filter_recurring_hash(covered_chars_kanji_plus, digital_font, 256))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

        for c in tqdm(covered_chars_kanji_plus, total=len(covered_chars_kanji_plus)):
            render_char = draw_func(c, digital_font, 256, padding=padding)
            if render_char is None:
                continue
            render_hash = hash(render_char.tobytes())
            if render_hash in filter_hashes:
                continue
            char_dir = os.path.join(save_path, str(ord(c)))
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
            if nonsquare:
                render_char.save(os.path.join(char_dir, f'{hex(ord(c))}_{idx}_{font_name}.png'))
            else:
                render_char.resize((64,64)).save(os.path.join(char_dir, f'{hex(ord(c))}_{idx}_{font_name}.png'))
            idx += 1


def paired_chars(dir_paths, save_path, omit="", nonsquare=False):

    idx = 0
    for dir_path in dir_paths:
        print(dir_path)
        for fpath in tqdm(glob.glob(os.path.join(dir_path, '*.png'))):
            c = ext_fname(fpath).split("_")[-1]
            if c.startswith("0x"):
                c = chr(int(c, base=16))
            if c in omit:
                continue
            char_dir = os.path.join(save_path, str(ord(c)))
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
            if nonsquare:
                Image.open(fpath).save(os.path.join(char_dir, f'PAIRED_{ext_fname(fpath)}_{idx}.png'))
            else:
                Image.open(fpath).resize((224,224)).save(os.path.join(char_dir, f'PAIRED_{ext_fname(fpath)}_{idx}.png'))
            idx += 1
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--crops_dir", type=str, required=True,
        help="Path to directory of character crops")
    parser.add_argument("--font_dir", type=str, default="./japan_font_files",
        help="Path to directory of fonts of interest for renders")
    parser.add_argument("--charset_dir", type=str, default="./japan_charsets",
        help="Path to directory of text files of character sets of interest")
    parser.add_argument("--save_dir", type=str, required=True,
        help="Save results to this directory")
    parser.add_argument("--exclude_fonts", type=str, default=None,
        help="Exclude particular fonts in `font_dir`")
    parser.add_argument("--padding", type=float, default=0.05,
        help="Add padding to renders, by percentage of height/width")
    parser.add_argument("--nonsquare", action="store_true", default=False,
        help="Allow crops/renders to be non-square, recommended")
    args = parser.parse_args()

    # get charset files and font files

    charset_files = glob.glob(os.path.join(args.charset_dir, '*'))
    font_files = glob.glob(os.path.join(args.font_dir, '*'))
    if not args.exclude_fonts is None:
        font_files = [ff for ff in font_files if not any(ef in ff for ef in args.exclude_fonts.split(","))]

    print(f"Fonts being used: {font_files}")

    # collect all chars of interest

    all_chars = []
    for csf in charset_files:
        charset_info = load_chars(csf)
        chars = [x[-1] for x in charset_info]
        all_chars.extend(chars)

    # harmonize character sets, save a reference list
    
    if "japan" in args.charset_dir:

        digits = list("0123456789")
        latin = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        extra_chars = list("靑鄉查々〇、)(,.")
        chars_to_remove = list("ッョカヵㇽ") + [chr(int("0x2f852", base=16))]
        full_charset = sorted(list(set(all_chars + digits + extra_chars + latin) - set(chars_to_remove)))
        with open("full_charset_jp.txt", "w") as f:
            f.write("\n".join(str(ord(c)) for c in full_charset))

    else:

        full_charset = sorted(list(set(all_chars)))
        with open("full_charset_en.txt", "w") as f:
            f.write("\n".join(str(ord(c)) for c in full_charset))

    # create draw function for renders

    if "eng" in args.charset_dir:
        draw_func = draw_single_char_ascender
        print("Using draw function with ascender space...")
    else:
        draw_func = draw_single_char

    print(f"Len all chars: {len(full_charset)}")

    # unified crop and render training
    
    render_chars(font_paths=font_files, unicode_chars=full_charset, 
        save_path=args.save_dir, padding=args.padding, draw_func=draw_func, nonsquare=args.nonsquare)
    paired_chars(dir_paths=args.crops_dir.split(","), save_path=args.save_dir, nonsquare=args.nonsquare)
    