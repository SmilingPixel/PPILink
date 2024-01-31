python3 run.py \
    --do_train \
    --nlp_model_name_or_path ../models/bert-small/ \
    --code_model_name_or_path ../models/code-bert-base/ \
    --train_batch_size 64 \
    --output_dir ../output \
    --train_file ../data/small_dataset/android_closed_issues_2011-01-01_2021-01-01_cluster_range_0-6999_preprocessed.json \
    --num_train_epochs 24 \
    --warmup_steps 4 \
    --save_steps 6 \
    --device cuda
    