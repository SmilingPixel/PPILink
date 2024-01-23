python3 run.py \
    --do_train \
    --model_name_or_path ../models/bert-small/ \
    --output_dir ../output \
    --train_file ../data/cluster_1024/android_closed_issues_2011-01-01_2021-01-01_all_clean_issues_NOT_CLEANED_1024_clusters.json \
    --device cuda