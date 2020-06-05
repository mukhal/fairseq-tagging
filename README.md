# Fairseq-tagging

![](logo/logo.png)
a Fairseq :fork_and_knife: fork adapted for sequence tagging/labeling tasks (NER, POS Tagging, etc) 


## Getting Started 

### 1. Prepare Data

Prepare your data is in the following IOB format: 

```
SOCCER NN B-NP O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O

CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

```
with `train.txt`, `valid.txt` and `test.txt` in `path/to/data/`

```
python preprocess.py --seqtag-data-dir path/to/data/ \
      --destdir path/to/data/bin \
      --nwordssrc 30000 \
      --bpe sentencepiece \
      --sentencepiece-model /path/to/sentencepiece.bpe.model
```

This converts data into `.source` and `.target` format and saves results to `path/to/data/fseq-outputs`:

#### sample converted data:

source:
```
SOCCER JAPAN GET LUCKY WIN ,
CHINA IN SURPRISE DEFEAT
SOCCER JAPAN GET LUCKY WIN
```
target:
```
O B-LOC O O O O
B-PER O O O
O B-LOC O O O

```


### 2. Train 
Let's train a tiny BERT (L=2, D=128, H=2) model from scratch:

```
python train.py data/conll-2003/bin \ 
      --arch bert_sequence_tagger_tiny \
      --criterion sequence_tagging \
      --max-sentences 16  \
      --task sequence_tagging \
      --max-source-positions 128 \
      -s source.bpe \
      -t target.bpe \
      --no-epoch-checkpoints \
      --lr 0.005 \
      --optimizer adam \
      --clf-report \
      --max-epoch 20 \
      --best-checkpoint-metric F1-score \
      --maximize-best-checkpoint-metric

```

### Results

## Tasks

- [x] log F1 metric on validation using Seqeva
- [x] save best model on validation data according to F1 score not loss
- [ ] `predict.py` script
- [x] work with BPE
- [x] load and finetune pretrained BERT or RoBERTa 
- [ ] LSTM models


