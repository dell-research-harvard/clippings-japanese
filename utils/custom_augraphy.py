###Custom augraphy pipeline
from augraphy import *
import cv2
import numpy as np
from numpy import asarray
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random


'''
Install the newest Augraphy !pip install git+https://github.com/sparkfish/augraphy.


Our Augraphy pipeline keeps most options used by the default pipeline.
We modify the default pipeline's options and parameters in the following way:
(1)We do not use Geometric in post phase
 because we don't want rotation upside down and already use torchvision for varying skewness.
(2)We do not use Bookbiding in post phase
 because it will cause huge image distortion such that no downstream tasks can be applicable. 
(3) We shift faxify to the first layout of post-phase because we find that the faxify effect in later layouts tends to amplfy 
other post phase noises, which may cause serious image corruption. 
and 
(4)For vertical text tasks, we slightly modify the gradient_width of folding effects from (0.1, 0.2) to (0.05, 0.1) 
because the width of vertical text images is much smaller than their horizontal counterparts. 
We keep the same default parameter range for horizontal cases. 


'''


#We first degine four phases: ink_phase, paper_phase, post_phase_h, post_phase_v

#Both horizontal and vertical text use the same ink phase
#The ink_phase is the same with Augraphy's default pipeline's ink_phase
ink_phase = [
        Dithering(
            dither=random.choice(["ordered", "floyd-steinberg"]),
            order=random.randint(3, 5),
            p=0.33,
        ),
        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 16),
            kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
            severity=(0.4, 0.6),
            p=0.33,
        ),
        BleedThrough(
            intensity_range=(0.1, 0.3),
            color_range=(32, 224),
            ksize=(17, 17),
            sigmaX=1,
            alpha=random.uniform(0.1, 0.2),
            offsets=(10, 20),
            p=0.33,
        ),
        Letterpress(
            n_samples=(100, 400),
            n_clusters=(200, 400),
            std_range=(500, 3000),
            value_range=(150, 224),
            value_threshold_range=(96, 128),
            blur=1,
            p=0.33,
        ),
        OneOf(
            [
                LowInkRandomLines(
                    count_range=(5, 10),
                    use_consistent_lines=random.choice([True, False]),
                    noise_probability=0.1,
                ),
                LowInkPeriodicLines(
                    count_range=(2, 5),
                    period_range=(16, 32),
                    use_consistent_lines=random.choice([True, False]),
                    noise_probability=0.1,
                ),
            ],
        ),
    ]

#Both horizontal and vertical text use the same paper phase
#The paper_phase is the same with Augraphy's default pipeline's paper_phase
paper_phase = [
        PaperFactory(p=0.33),
        ColorPaper(
            hue_range=(0, 255),
            saturation_range=(10, 40),
            p=0.33,
        ),
        WaterMark(
            watermark_word="random",
            watermark_font_size=(10, 15),
            watermark_font_thickness=(20, 25),
            watermark_rotation=(0, 360),
            watermark_location="random",
            watermark_color="random",
            watermark_method="darken",
            p=0.33,
        ),
        OneOf(
            [
                AugmentationSequence(
                    [
                        NoiseTexturize(
                            sigma_range=(3, 10),
                            turbulence_range=(2, 5),
                        ),
                        BrightnessTexturize(
                            texturize_range=(0.9, 0.99),
                            deviation=0.03,
                        ),
                    ],
                ),
                AugmentationSequence(
                    [
                        BrightnessTexturize(
                            texturize_range=(0.9, 0.99),
                            deviation=0.03,
                        ),
                        NoiseTexturize(
                            sigma_range=(3, 10),
                            turbulence_range=(2, 5),
                        ),
                    ],
                ),
            ],
            p=0.33,
        ),
        Brightness(
            brightness_range=(0.9, 1.1),
            min_brightness=0,
            min_brightness_value=(120, 150),
            p=0.1,
        ),
    ]

# Horizontal text use this post phase
post_phase_h = [
        OneOf(
            [
                PageBorder(
                    side="random",
                    border_background_value=(230, 255),
                    flip_border=random.choice([0, 1]),
                    width_range=(5, 30),
                    pages=None,
                    noise_intensity_range=(0.3, 0.8),
                    curve_frequency=(2, 8),
                    curve_height=(2, 4),
                    curve_length_one_side=(50, 100),
                    value=(32, 150),
                    same_page_border=random.choice([0, 1]),
                ),
                DirtyRollers(
                    line_width_range=(2, 32),
                    scanline_type=0,
                ),
                Faxify(
                    scale_range=(0.3, 0.6),
                    monochrome=random.choice([0, 1]),
                    monochrome_method="random",
                    monochrome_arguments={},
                    halftone=random.choice([0, 1]),
                    invert=1,
                    half_kernel_size=random.choice([(1, 1), (2, 2)]),
                        angle=(0, 360),
                    sigma=(1, 3),
                
                ),
            ],
            p=0.33,
        ),
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
            ],
            p=0.33,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.33,
        ),
        SubtleNoise(
            subtle_range=random.randint(5, 10),
            p=0.33,
        ),
        Jpeg(
            quality_range=(25, 95),
            p=0.33,
        ),
        Folding(
            fold_x=None,
            fold_deviation=(0, 0),
            fold_count=random.randint(1, 6),
            fold_noise=random.uniform(0, 0.2),
            gradient_width=(0.1, 0.2),
            gradient_height=(0.01, 0.02),
            p=0.33,
        ),
        Markup(
            num_lines_range=(2, 7),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(1, 2),
            markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
            markup_color="random",
            single_word_mode=False,
            repetitions=1,
            p=0.33,
        ),
        PencilScribbles(
            size_range=(100, 800),
            count_range=(1, 6),
            stroke_count_range=(1, 2),
            thickness_range=(2, 6),
            brightness_change=random.randint(64, 224),
            p=0.33,
        ),
        BadPhotoCopy(
            mask=None,
            noise_type=-1,
            noise_side="random",
            noise_iteration=(1, 2),
            noise_size=(1, 3),
            noise_value=(128, 196),
            noise_sparsity=(0.3, 0.6),
            noise_concentration=(0.1, 0.6),
            blur_noise=random.choice([True, False]),
            blur_noise_kernel=random.choice([(3, 3), (5, 5), (7, 7)]),
            wave_pattern=random.choice([True, False]),
            edge_effect=random.choice([True, False]),
            p=0.33,
        ),
        Gamma(
            gamma_range=(0.9, 1.1),
            p=0.33,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type="random",
            ntimes=(2, 6),
            nscales=(0.9, 1.0),
            edge="random",
            edge_offset=(10, 50),
            use_figshare_library=0,
            p=0.33,
        ),
        
    ]

# Vertical text use this post phase
post_phase_v = [
        OneOf(
            [
                PageBorder(
                    side="random",
                    border_background_value=(230, 255),
                    flip_border=random.choice([0, 1]),
                    width_range=(5, 30),
                    pages=None,
                    noise_intensity_range=(0.3, 0.8),
                    curve_frequency=(2, 8),
                    curve_height=(2, 4),
                    curve_length_one_side=(50, 100),
                    value=(32, 150),
                    same_page_border=random.choice([0, 1]),
                ),
                DirtyRollers(
                    line_width_range=(2, 32),
                    scanline_type=0,
                ),
                Faxify(
                    scale_range=(0.3, 0.6),
                    monochrome=random.choice([0, 1]),
                    monochrome_method="random",
                    monochrome_arguments={},
                    halftone=random.choice([0, 1]),
                    invert=1,
                    half_kernel_size=random.choice([(1, 1), (2, 2)]),
                        angle=(0, 360),
                    sigma=(1, 3),
                
                ),
            ],
            p=0.33,
        ),
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
            ],
            p=0.33,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.33,
        ),
        SubtleNoise(
            subtle_range=random.randint(5, 10),
            p=0.33,
        ),
        Jpeg(
            quality_range=(25, 95),
            p=0.33,
        ),
        #We modify gradient_width
        Folding(
            fold_x=None,
            fold_deviation=(0, 0),
            fold_count=random.randint(1, 6),
            fold_noise=random.uniform(0, 0.2),
            gradient_width=(0.05, 0.1), 
            gradient_height=(0.01, 0.02),
            p=0.33,
        ),
        Markup(
            num_lines_range=(2, 7),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(1, 2),
            markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
            markup_color="random",
            single_word_mode=False,
            repetitions=1,
            p=0.33,
        ),
        PencilScribbles(
            size_range=(100, 800),
            count_range=(1, 6),
            stroke_count_range=(1, 2),
            thickness_range=(2, 6),
            brightness_change=random.randint(64, 224),
            p=0.33,
        ),
        BadPhotoCopy(
            mask=None,
            noise_type=-1,
            noise_side="random",
            noise_iteration=(1, 2),
            noise_size=(1, 3),
            noise_value=(128, 196),
            noise_sparsity=(0.3, 0.6),
            noise_concentration=(0.1, 0.6),
            blur_noise=random.choice([True, False]),
            blur_noise_kernel=random.choice([(3, 3), (5, 5), (7, 7)]),
            wave_pattern=random.choice([True, False]),
            edge_effect=random.choice([True, False]),
            p=0.33,
        ),
        Gamma(
            gamma_range=(0.9, 1.1),
            p=0.33,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type="random",
            ntimes=(2, 6),
            nscales=(0.9, 1.0),
            edge="random",
            edge_offset=(10, 50),
            use_figshare_library=0,
            p=0.33,
        ),
        
    ]    

   


#We define two pipeline, one for horizontal and the other for vertcial case.
#To see each layout's output, modify save_outputs into TRUE
#To see log file, turn log into TRUE

###if image is horizontal, use pipeline_h
###if image is vertical, use pipeline_v



pipeline_h    = AugraphyPipeline(ink_phase, paper_phase, post_phase_h, save_outputs=False, log=False)


pipeline_v    = AugraphyPipeline(ink_phase, paper_phase, post_phase_v, save_outputs=False, log=False)


