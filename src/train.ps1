python run.py `
    --do_train `
    --nlnl_model_name_or_path ../models/bert-small/ `
    --nlpl_model_name_or_path ../models/code-bert-base/ `
    --train_batch_size 4 `
    --output_dir ../output `
    --train_file ../data/small_dataset/android_closed_issues_2011-01-01_2021-01-01_cluster_range_00000-11999_preprocessed.json `
    --num_train_epochs 1
    