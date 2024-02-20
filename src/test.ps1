python run.py `
    --do_test `
    --main_model_name_or_path ../output/ckpt/20240130064905/epoch_16 `
    --nlnl_tokenizer_name_or_path ../models/roberta-base/ `
    --nlpl_tokenizer_name_or_path ../models/codereviewer/ `
    --eval_batch_size 4 `
    --output_dir ../output `
    --test_file ../data/cluster_00000-17999/15000-17999_text_preprocessed_commits_filtered_preprocessed.json