###Data prep
# 1) Make synthetic place names data : scripts/vit_scripts/synth_line_H_V_skip_fonts_wordlist.py
# 2) Make train-val-test splits : scripts/vit_scripts/split_dataset.py
# 3) Make query folders : scripts/vit_scripts/make_query_folders.py

##Fine tuning data - also used for multimodal
# 1) scripts/vit_scripts/prep_image_folder_ft.py


#Visual Record Linkage - 20k placenames
##Pretraining
###No offline mining
CUDA_VISIBLE_DEVICES=1,3 python /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/clippings-japanese/train_vit.py --root_dir_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/images/ --train_images_dir /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/single_font_train/  --run_name vit_word_nohn_japan_center_places_3000_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 256 --num_epochs 10 --num_passes 1 --lr 0.00005791180952082007 --test_at_end --imsize 224 --train_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/splits/train.json" --val_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/splits/val.json" --test_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/splits/test.json" --m 8 --temp 0.048 --weight_decay 0.0398 --resize --epoch_viz_dir /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/ --use_renders

###Offline hn mining
CUDA_VISIBLE_DEVICES=0,1 python /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/clippings-japanese/train_vit.py --root_dir_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/images/ --train_images_dir /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/single_font_train/  --run_name vit_word_hn_japan_center_places_3000_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 256 --num_epochs 4 --num_passes 1 --lr 0.00005791180952082007 --test_at_end --imsize 224 --train_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/splits/train.json" --val_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/splits/val.json" --test_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/word_dump_centered_japan_places_3000/splits/test.json" --m 8 --temp 0.048 --weight_decay 0.0398 --resize --hns_txt_path ./vit_word_nohn_japan_center_places_3000_recheck/hns.txt --epoch_viz_dir /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz --use_renders

##Fine-tuning
###No offline mining
CUDA_VISIBLE_DEVICES=1,2 python /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/clippings-japanese/train_vit.py --root_dir_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/images/  --run_name vit_word_nohn_japan_center_places_20000_finetuned_new_test_val_test_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 252 --num_epochs 1 --num_passes 1 --lr 0.00002695 --test_at_end --imsize 224 --train_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train.json" --val_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/val.json" --test_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/test.json" --m 6 --temp 0.07347 --weight_decay 0.01737 --resize --epoch_viz_dir /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/  --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/vit_word_hn_japan_center_places_20000/enc_best.pth" --train_images_dir "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_images"

###Offline mining
CUDA_VISIBLE_DEVICES=1,2 python /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/clippings-japanese/train_vit.py --root_dir_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/images/  --run_name vit_word_nohn_japan_center_places_20000_finetuned_new_test_val_hn_test_recheck --auto_model_timm vit_base_patch16_224.dino --batch_size 252 --num_epochs 1 --num_passes 1 --lr 0.00002695 --test_at_end --imsize 224 --train_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train.json" --val_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/val.json" --test_ann_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/test.json" --m 6 --temp 0.07347 --weight_decay 0.01737 --resize --epoch_viz_dir /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit/epoch_viz/  --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/vit_word_hn_japan_center_places_20000/enc_best.pth" --hns_txt_path ./vit_word_nohn_japan_center_places_20000_finetuned_new_test_val/hns.txt --train_images_dir "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/vision_dir/vision_ft_corr_val/splits/train_images"



###Language model
###SBERT


###BiCLIP 
## Pretrain CLIP using synthetic + unlabeled data
##Prep data first

python clippings-japanese/train_clippings.py --clip_lr 5e-5 --train_data_type synth_unlabelled --wandb_name clip_pretrain_unlabelled_m1_v3_test_new_repo --training_type pretrain

###Pretrain biclip using synthetic data. Offline mining first
CUDA_VISIBLE_DEVICES=1 python clippings-japanese/train_clippings.py --clip_lr 5e-6 --train_data_type synth --wandb_name bienc_clip_pretrain_synth_m3_v3_hardneg --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt"


###Fine-tune biclip using labelled data. Offline mining first
CUDA_VISIBLE_DEVICES=3 python clippings-japanese/train_clippings.py --clip_lr 5e-6 --train_data_type labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_hardneg --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/bienc_clip_pretrain_synth_m3_v3_hardneg.pt"


##norm after average
CUDA_VISIBLE_DEVICES=2 python clippings-japanese/train_clippings.py --clip_lr 5e-7 --clip_weight_decay 0.05 --train_data_type labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_hardneg_norm_supcon05 --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.05 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/bienc_clip_pretrain_synth_m3_v3_hardneg.pt" 
##Try higher weight decay, lower lr



######BICLIP ONLY VISION

###Pretrain biclip using synthetic data. Offline mining first
CUDA_VISIBLE_DEVICES=1 python clippings-japanese/train_clippings.py --clip_lr 5e-6 --train_data_type synth --wandb_name bienc_clip_pretrain_synth_m3_v3_hardneg_img_only --m 3 --training_type train_bienc --im_wt 1 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt"


###Fine-tune biclip using labelled data. Offline mining first
CUDA_VISIBLE_DEVICES=3 python clippings-japanese/train_clippings.py --clip_lr 5e-6 --train_data_type labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_hardneg_img_only --m 3 --training_type train_bienc --im_wt 1 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/bienc_clip_pretrain_synth_m3_v3_hardneg_img_only.pt"



######BICLIP ONLY Texxt

###Pretrain biclip using synthetic data. Offline mining first
CUDA_VISIBLE_DEVICES=1 python clippings-japanese/train_clippings.py --clip_lr 5e-6 --train_data_type synth --wandb_name bienc_clip_pretrain_synth_m3_v3_hardneg_text_only --m 3 --training_type train_bienc --im_wt 0 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt"


###Fine-tune biclip using labelled data. Offline mining first
CUDA_VISIBLE_DEVICES=3 python clippings-japanese/train_clippings.py --clip_lr 5e-6 --train_data_type labelled --wandb_name bienc_clip_pretrain_synth_m3_v3_hardneg_text_only --m 3 --training_type train_bienc --im_wt 0 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/bienc_clip_pretrain_synth_m3_v3_hardneg_text_only.pt"



###Inference
##On supervised model
###Inferpr tk
############Effocr

##Only text
#CUDA_VISIBLE_DEVICES=1 python clippings-japanese/clipper_inference.py  --pooling_type "text" --output_prefix sup_text_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_0bienc_clip_pretrain_synth_m3_v3_hardneg_text_only.pt --ocr_result "effocr" --infer_partnertk

##Only image
#CUDA_VISIBLE_DEVICES=2 python clippings-japanese/clipper_inference.py  --pooling_type "image" --output_prefix sup_img_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_0bienc_clip_pretrain_labelled_m3_v3_hardneg_img_only.pt --ocr_result "effocr" --infer_partnertk

##Pooled 
#CUDA_VISIBLE_DEVICES=0 python clippings-japanese/clipper_inference.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_hardneg_norm_final.pt --ocr_result "effocr" --infer_partnertk

###################

##Only text
CUDA_VISIBLE_DEVICES=1 python clippings-japanese/clipper_inference.py  --pooling_type "text" --output_prefix sup_text_gcv  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_0bienc_clip_pretrain_synth_m3_v3_hardneg_text_only.pt --ocr_result "gcv" --infer_partnertk

##Only image
# CUDA_VISIBLE_DEVICES=2 python clippings-japanese/clipper_inference.py  --pooling_type "image" --output_prefix sup_img_gcv  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_0bienc_clip_pretrain_labelled_m3_v3_hardneg_img_only.pt --ocr_result "gcv" --infer_partnertk

##Pooled 
#CUDA_VISIBLE_DEVICES=3 python clippings-japanese/clipper_inference.py  --pooling_type "mean" --output_prefix mean_norm_1_gcv --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_hardneg_norm_final.pt --ocr_result "gcv" --infer_partnertk






###On self-supervised model
##Text only
#CUDA_VISIBLE_DEVICES=1 python clippings-japanese/clipper_inference.py  --pooling_type "text" --output_prefix selfsup_text_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt --ocr_result "effocr" --infer_partnertk 

###Image only
#CUDA_VISIBLE_DEVICES=1 python clippings-japanese/clipper_inference.py  --pooling_type "image" --output_prefix selfsup_img_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt --ocr_result "effocr" --infer_partnertk

###Mean
#CUDA_VISIBLE_DEVICES=2 python clippings-japanese/clipper_inference.py  --pooling_type "mean" --output_prefix selfsup_mean_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt --ocr_result "effocr" --infer_partnertk


###GCV pooled self supervised

###text only
CUDA_VISIBLE_DEVICES=1 python clippings-japanese/clipper_inference.py  --pooling_type "text" --output_prefix selfsup_text_gcv  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt --ocr_result "gcv" --infer_partnertk

##Mean 
# CUDA_VISIBLE_DEVICES=3 python clippings-japanese/clipper_inference.py  --pooling_type "mean" --output_prefix selfsup_mean_gcv  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/epoch_40clip_pretrain_unlabelled_m1_v3.pt --ocr_result "gcv" --infer_partnertk



###NOW INFERENCE FOR BEST CLIPPER MODELS - only on test data not full prtk

#Only text
CUDA_VISIBLE_DEVICES=1 python clippings-japanese/clipper_inference.py  --pooling_type "text" --output_prefix sup_text_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_0bienc_clip_pretrain_synth_m3_v3_hardneg_text_only.pt --ocr_result "effocr" 
#Only image
CUDA_VISIBLE_DEVICES=2 python clippings-japanese/clipper_inference.py  --pooling_type "image" --output_prefix sup_img_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_0bienc_clip_pretrain_labelled_m3_v3_hardneg_img_only.pt --ocr_result "effocr" 

#Pooled 
CUDA_VISIBLE_DEVICES=3 python clippings-japanese/clipper_inference.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/best_models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_hardneg_norm_final.pt --ocr_result "effocr" 
