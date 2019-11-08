### Training

To train a GPT2 model on MCScript, NarrativeQA, SocialIQA, and COSMOSQA, run:
```
python gpt2/train_gpt2.py
```

To train a BERT model on QUOREF, run:
```
python bert/pytorch-pretrained-BERT/examples/run_squad.py \
    --bert_model bert-base-uncased \
    --output_dir ./bert/models/quoref \
    --train_file raw_data/quoref/quoref-train-v0.1.json \
    --gradient_accumulation_steps 8 \
    --do_train
```

To train a BERT model on ROPES, run:
```
python bert/pytorch-pretrained-BERT/examples/run_squad.py \
    --bert_model bert-base-uncased \
    --output_dir ./bert/models/ropes \
    --train_file raw_data/quoref/train-v0.4.json \
    --gradient_accumulation_steps 8 \
    --do_train
```

To train a Numerically Augmented QANET on DROP, run:
```
allennlp train naqanet/drop_config.json -s naqanet/models/drop/ 
```

To train a Numerically Augmented BERT (NABERT+) on DROP, run:
```
allennlp train nabert/drop_config.json --include-package nabert -s nabert/models/drop/
```

### Predictions
To get predictions on MCScript, NarrativeQA, SocialIQA, and COSMOSQA from a trained GPT2 model run:
```
python gpt2/generate_gpt2.py
```

To get predictions on QUOREF from a trained BERT model, run :
```
python pytorch-pretrained-BERT/examples/run_squad.py \
    --bert_model bert-base-uncased \
    --output_dir ./bert/models/quoref \
    --predict_file raw_data/quoref/quoref-<data_type>-v0.1.json \
    --n_best_size <top n answers to return> \
    --do_predict
```

To get predictions on ROPES from a trained BERT model, run :
```
python pytorch-pretrained-BERT/examples/run_squad.py \
    --bert_model bert-base-uncased \
    --output_dir ./bert/models/ropes \
    --predict_file raw_data/ropes/<data_type>-v0.4.json \
    --n_best_size <top n answers to return> \
    --do_predict
```

To get predictions on DROP from a trained NAQANET model, run:
```
python naqanet/drop_predict.py -d <device_num> -w <weights_file>
```

To get predictions on DROP from a trained NABERT+ model, run:
```
python nabert/drop_predict.py -d <device_num> -w <weights_file>
```

### Creating Hits
To create HITS on the datasets, run the following dataset-specific commands

```
python mt_html/create_hits.py \
    --html mt_html/narrativeqa.html \
    --csv merge_predictions/to_label/narrativeqa.csv \
    --out mt_html/narrativeqa/ \
    --num_hits 10 

python mt_html/create_hits.py \
    --html mt_html/mcscript.html \
    --csv merge_predictions/to_label/mcscript.csv \
    --out mt_html/mcscript/ \
    --num_hits 10 
```

Then merge the links into a single file
```
python mt_html/generate_html_links.py
```