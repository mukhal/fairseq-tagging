# Fairseq-tagger

a Fairseq fork for sequence tagging/labeling tasks (NER, PSS Tagging, etc)




### Getting Started 

#### 1. Prepare Data
```
preprocess.py --seqtag-data-dir data/conll-2003 \
      --destdir data/conll-2003/bin \
      --nwordssrc 30000 \
      --bpe sentencepiece \
      --sentencepiece-model /path/to/sentencepiece.bpe.model
```

#### 2. Train 
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


