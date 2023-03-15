# Levenshtein and Stroke Similarity
python scripts/rule_based_lev_fuzzyChineseStroke.py --lev --fuzzychinese_stroke --save_output ./rule_based_output

# SelfSup Visual Linking	(Clean/Noisy OCR)
infer_clippings.py  --pooling_type "image" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/only_pretrained_model.pt --ocr_result "effocr" 
infer_clippings.py  --pooling_type "image" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/only_pretrained_model.pt --ocr_result "gcv" 

# SelfSup Language Linking	(Clean/Noisy OCR)
infer_clippings.py  --pooling_type "text" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/only_pretrained_model.pt --ocr_result "effocr" 
infer_clippings.py  --pooling_type "text" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/only_pretrained_model.pt --ocr_result "gcv" 

# SelfSup Multimodal Linking	(Clean/Noisy OCR)
infer_clippings.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/only_pretrained_model.pt --ocr_result "effocr" 
infer_clippings.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/only_pretrained_model.pt --ocr_result "gcv" 


# Vit Visual Linking	- OCR does not matter
python infer_vit.py --root_folder "/path/toPaddleOCR_testing/Paddle_test_images/japan_vit_all_infer_prtkfinal_synthonly" --timm_model vit_base_patch16_224.dino  --checkpoint_path "/path/todeeprecordlinkage/vision_dir/best_models/best_vision_model.pth" --recopy

# Sup Visual Linking	(Clean/Noisy OCR)
infer_clippings.py  --pooling_type "image" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clippings_model_image_only.pt --ocr_result "effocr" 
infer_clippings.py  --pooling_type "image" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clippings_model_image_only.pt --ocr_result "gcv" 


# Sup Language Linking	(Clean/Noisy OCR)
infer_clippings.py  --pooling_type "text" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clippings_model_lang_only.pt --ocr_result "effocr" 
infer_clippings.py  --pooling_type "text" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clippings_model_lang_only.pt --ocr_result "gcv" 



# Sup Multimodal Linking	(Clean/Noisy OCR)
infer_clippings.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clippings_model_multimodal.pt --ocr_result "effocr" 
infer_clippings.py  --pooling_type "mean" --output_prefix mean_norm_1_effocr  --checkpoint_path /path/to/multimodal_record_linkage/best_models/clippings_model_multimodal.pt --ocr_result "gcv" 
