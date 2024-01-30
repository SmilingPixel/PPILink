python3 run.py \
    --do_test \
    --main_model_name_or_path ../output/ckpt/... \
    --eval_batch_size 64 \
    --output_dir ../output \
    --test_file ../data/small_dataset/android_closed_issues_2011-01-01_2021-01-01_cluster_range_8500-9999_preprocessed.json \
    --device cuda
    