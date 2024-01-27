python3 run.py \
    --do_train \
    --nlp_model_name_or_path ../models/bert-small/ \
    --code_model_name_or_path ../models/code-bert-base/ \
    --train_batch_size 128 \
    --output_dir ../output \
    --train_file ../data/small_dataset/android_closed_issues_2011-01-01_2021-01-01_cluster_range_0-6999_preprocessed.json \
    --device cuda
    