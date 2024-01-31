python3 run.py \
    --do_test \
    --main_model_name_or_path ../output/ckpt/20240130064905/epoch_16 \
    --nlp_model_name_or_path ../models/bert-small/ \
    --code_model_name_or_path ../models/code-bert-base/ \
    --eval_batch_size 64 \
    --output_dir ../output \
    --test_file ../data/small_dataset/android_closed_issues_2011-01-01_2021-01-01_cluster_range_12000-14999_preprocessed.json \
    --device cuda
    