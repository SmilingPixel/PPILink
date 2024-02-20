# PPILink
PPILink is a deep learning model aimed at recovering missing PR-Issue links. It's trained on over 10,000 links between pull-requests and issues in GitHub.

The model checkpoint will be released soon.


## Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

Our dataset is based on the PI-Link dataset, which is a ground-truth dataset of links between pull-requests and issues in GitHub. It is available at [PI-Link: A Ground-Truth Dataset of Links Between Pull-Requests and Issues in GitHub](https://ieeexplore.ieee.org/abstract/document/10002372/)

We will release our dataset soon.


## Training

For Linux:
```bash
cd src && bash train.sh
```

For Windows powershell:
```bash
cd src && .\train.ps1
```

Below is a demo training script (train.sh):

```bash
python3 run.py \
    --do_train \
    --do_eval \
    --nlnl_model_name_or_path ../models/roberta-base/ \
    --nlpl_model_name_or_path ../models/codereviewer/ \
    --nlnl_tokenizer_name_or_path ../models/roberta-base/ \
    --nlpl_tokenizer_name_or_path ../models/codereviewer/ \
    --learning_rate 1e-5 \
    --train_batch_size 8 \
    --output_dir ../output \
    --train_file ../data/YOUR_TRAIN_DATA.json \
    --eval_file ../data/YOUR_EVAL_DATA.json \
    --num_train_epochs 18 \
    --warmup_steps 4 \
    --save_steps 6 \
    --device cuda
```

## Testing

For Linux:
```bash
cd src && bash test.sh
```

For Windows powershell:
```bash
cd src && .\test.ps1
```

Below is a demo testing script (test.sh):

```bash
python3 run.py \
    --do_test \
    --main_model_name_or_path ../output/ckpt/20240130064905/epoch_16 \
    --nlnl_tokenizer_name_or_path ../models/roberta-base/ \
    --nlpl_tokenizer_name_or_path ../models/codereviewer/ \
    --eval_batch_size 64 \
    --output_dir ../output \
    --test_file ../data/cluster_00000-17999/15000-17999_text_preprocessed_commits_filtered_preprocessed.json \
    --device cuda
```
