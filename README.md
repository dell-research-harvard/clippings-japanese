
# CLIPPINGS (Japanese)

This repo is for CLIPPINGS for record linkage (japanese). It uses the CLIP [model](https://github.com/rinnakk/japanese-clip) trained by a group that has no connection to this project. 

## Repo structure


- sample_data : Contains sample data
    -  source_images : directory containing source images
    -  target images : directory containing target images
    -  source_ocr_data.json  : {image:OCR_text}
    -  target_ocr_data.json : {image:OCR_text}
    -  ground_truth.csv : |source_path|target_path|

- datasets
    - clippings_data_loaders.py : Contains the pytorch datasets, dataloaders and miners neccesary for training CLIPPINGS
    - vit_datasets.py : pytorch custom datasets/loaders for training and working with ViT models 
    - vit_samplers.py : pytorch custom samplers for use in training

- japan_font_files : A directory containing fonts needed to generate synthetic renders for training CLIPPINGS and ViT

- models
    - encoders.py: Contains some model components used in ViT (also an MLP for use in CLIPPINGS instead of mean pooling - deprecated)

- scripts
    - pre_process_data.py : To prepare the data needed for training Japanese CLIPPINGS for multi-modal record linkage
    - prep_image_folder_ft.py : To prepare LABELLED data for training ViT and splitting into train-test-val (the same splits used for multimodal models)
    - gen_synthetic_dataset
        - create_font_image_folder.py
        - multimodal_synth_noise.py
    - match_nomatch
        - fill this up
    - vit_scripts (only listing important ones)
        - synth_line_H_V_skip_fonts_wordlist.py : Generate synthetic text redners using a word list
        - split_dataset.py : Split synthetically rendered data into train-val-test

- utils 
    - datasets_utils.py : Contains the main torchvision transformations needed to transform the images before feeding them into the model
    - gen_synthetic_segments : Some data augmentation functions to prepare random augmentations (Augmentations are only used in the ViT training and not the CLIPPINGS model)

- train_clippings.py : Script that supports both language-image pretraining of an underlying clip model as well as the main function to train "CLIPPINGS"

- infer_clippings.py : Run inference to embed image-text pairs given model weights, perform SLINK, find the optimum threshold using the val set and present ARI for the test data

- requirements.yaml : The conda environment containing all dependencies

## Code usage
This section provides the commands (with relevant arguments) to replicate the results in the main paper. 

### CLIPPINGS
Use relevant hyperparameters from Table X in the supplementary material file

#### Train 

Pretrain the base CLIP model (example)

```
python train_clippings.py --clip_lr 5e-5 --train_data_type synth_unlabelled --wandb_name clip_pretrain_unlabelled_m1_v3 --training_type pretrain

```

Train the CLIPPINGS model (example)

First train using synthetic data

```
python train_clippings.py --clip_lr 5e-6 --train_data_type synth --wandb_name bienc_clip_pretrain_synth_m3_v3_hardneg --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/path/to/pretrainedclip/clip_pretrain_unlabelled_m1_v3.pt"
```

Now train using labelled data
```
python train_clippings.py --clip_lr 5e-7 --clip_weight_decay 0.05 --train_data_type labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_hardneg_norm --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.05 --train_hardneg --checkpoint "/path/to/pretrainedclip/bienc_clip_pretrain_synth_m3_v3_hardneg.pt"

```

For image-only training, use im_wt = 1 and for language-only training, use im_wt=0. 



#### Inference

``` 
infer_clippings.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_hardneg_norm_final.pt --ocr_result "effocr" 
```
--pooling_type can be either "mean", "text" or "image" and --ocr_result can be either "effocr" or "gcv". effocr corresponds to the "clean" ocr and "gcv" corresponds to the "noisy" ocr. 






### Vision Transformer

#### Train

Synthetic contrastive training

No offline hard negative mining

```
 python train_vit.py --root_dir_path /path/to/word_dump_centered_japan_places_20000/images/ --train_images_dir /path/to/word_dump_centered_japan_places_20000/single_font_train/  --run_name vit_word_nohn_japan_center_places_20000_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 256 --num_epochs 10 --num_passes 1 --lr 0.00005791180952082007 --test_at_end --imsize 224 --train_ann_path "/path/to/word_dump_centered_japan_places_20000/splits/train.json" --val_ann_path "/path/to/word_dump_centered_japan_places_20000/splits/val.json" --test_ann_path "/path/to/word_dump_centered_japan_places_20000/splits/test.json" --m 8 --temp 0.048 --weight_decay 0.0398 --resize --epoch_viz_dir /path/toPaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/ --use_renders

```

With offline hard negative mining

```
python train_vit.py --root_dir_path /path/to/word_dump_centered_japan_places_20000/images/ --train_images_dir /path/to/word_dump_centered_japan_places_20000/single_font_train/  --run_name vit_word_hn_japan_center_places_20000_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 256 --num_epochs 4 --num_passes 1 --lr 0.00005791180952082007 --test_at_end --imsize 224 --train_ann_path "/path/to/word_dump_centered_japan_places_20000/splits/train.json" --val_ann_path "/path/to/word_dump_centered_japan_places_20000/splits/val.json" --test_ann_path "/path/to/word_dump_centered_japan_places_20000/splits/test.json" --m 8 --temp 0.048 --weight_decay 0.0398 --resize --hns_txt_path ./vit_word_nohn_japan_center_places_20000_recheck/hns.txt --epoch_viz_dir /path/toPaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz --use_renders

```


Fine-tuning on labelled data


No offline hard negative mining

```
 python train_vit.py --root_dir_path /path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/images/  --run_name vit_word_nohn_japan_center_places_20000_finetuned_new_test_val_test_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 252 --num_epochs 1 --num_passes 1 --lr 2e-6 --test_at_end --imsize 224 --train_ann_path "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train.json" --val_ann_path "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/val.json" --test_ann_path "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/test.json" --m 3 --temp 0.09 --weight_decay 0.1 --resize --epoch_viz_dir /path/toPaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/  --checkpoint "/path/to/vit_word_hn_japan_center_places_20000/enc_best.pth" --train_images_dir "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_images"

```

With offline hard negative mining

```
 python train_vit.py --root_dir_path /path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/images/  --run_name vit_word_nohn_japan_center_places_20000_finetuned_new_test_val_hn_test_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 252 --num_epochs 1 --num_passes 1 --lr 2e-6 --test_at_end --imsize 224 --train_ann_path "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train.json" --val_ann_path "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/val.json" --test_ann_path "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/test.json" --m 3 --temp 0.09 --weight_decay 0.1 --resize --epoch_viz_dir /path/toPaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/  --checkpoint "/path/to/vit_word_hn_japan_center_places_20000/enc_best.pth" --hns_txt_path ./vit_word_nohn_japan_center_places_20000_finetuned_new_test_val/hns.txt --train_images_dir "/path/todeeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_images"

```

#### Inference

```
python infer_vit.py --root_folder "/path/toPaddleOCR_testing/Paddle_test_images/japan_vit_all_infer_prtkfinal_synthonly" --timm_model vit_base_patch16_224.dino  --checkpoint_path "/path/todeeprecordlinkage/vision_dir/best_models/enc_best_e_ulti.pth" --recopy

```


### Rule-based traditional Record-linkage baseline
This script will replicate all results related to our rule-based baseline. 

```
python scripts/rule_based_lev_fuzzyChineseStroke.py --lev --fuzzychinese_stroke --save_output ./rule_based_output
```


### Creating network plots
We have scripts to generate the network figures in the paper. Refer to docs/network_vis_pipeline.md to regenerate the figures with appropriate seeds already filled in.

### Synthetic data generation pipeline 
We also have scripts to generate synthetic data (both image-only and image-text versions). 
Refer to docs/synthetic_data_generation.md for details


### Replication of main results

|                                | **Noisy OCR** | **Clean OCR**    |
|--------------------------------|---------------|------------------|
| **Levenshtein distance**       | 0.630         | 0.731            |
| **Stroke n-gram similarity**   | 0.689         | 0.731            |
| **SelfSup Visual Linking**     | 0.769         | 0.769            |
| **SelfSup Language Linking**   | 0.740         | 0.790            |
| **SelfSup Multimodal Linking** | 0.845         | 0.849            |
| **Sup Visual Linking**         | 0.878         | 0.878            |
| **Sup Visual Linking**         | 0.924         | 0.924            |
| **Sup Language Linking**       | 0.790         | 0.882            |
| **Sup Multimodal Linking**     | 0.937         | 0.945            |

Run the following file:
```
results_replication.sh
```
