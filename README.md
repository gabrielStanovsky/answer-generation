# Candidate answer generation

### Training Models for Answer generation

##### Training

To train a GPT2 model on MCScript, NarrativeQA, SocialIQA, and COSMOSQA, run:
```
python gpt2/train_gpt2.py
```

To train the Huggingfaces GPT2 models where we only calculate the loss on the answer tokens, run:
```
allennlp train huggingface_gpt2/narrativeqa_config.json \
    --include-package huggingface_gpt2\
    -s huggingface_gpt2/models/narrativeqa

allennlp train huggingface_gpt2/mcscript_config.json \
    --include-package huggingface_gpt2\
    -s huggingface_gpt2/models/mcscript

allennlp train huggingface_gpt2/socialiqa_config.json \
    --include-package huggingface_gpt2\
    -s huggingface_gpt2/models/socialiqa

allennlp train huggingface_gpt2/cosmosqa_config.json \
    --include-package huggingface_gpt2\
    -s huggingface_gpt2/models/cosmosqa
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

##### Predictions
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

### Backtranslation
To create backtranslations of the generative QA datasets (COSMOSQA, NarrativeQA, MCScript, and SocialIQA), run
```
python backtranslation/backtranslate_answers.py
```
which creates backtranslations in the `backtranslation/<dataset>` directory. 

### Masked Answer Generation
To create answer candidates using masked answer fill-in, run:
```

```

# Merging answer candidate sources
Given that we have candidate answers from many different sources, we now have to merge them together. 
Each dataset has its own merging file. The run commmand for all datasets is: 
```
python merge_predictions/merge_cosmosqa.py
python merge_predictions/merge_drop.py
python merge_predictions/merge_mcscript.py
python merge_predictions/merge_narrativeqa.py
python merge_predictions/merge_quoref.py
python merge_predictions/merge_ropes.py
python merge_predictions/merge_utils.py
```
This puts the merged answer candidates into `merge_predictions/to_label`.

# Creating Hits
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