
<img src="logo/logo.png" width="500" height="130">

   a [Fairseq](https://github.com/pytorch/fairseq) fork :fork_and_knife: adapted for sequence tagging/labeling tasks (NER, POS Tagging, etc) 

## Motivation
Fairseq is a great tool for training seq2seq models. However, it was not meant for sequence tagging tasks such as Ner or PoS tagging, etc. This should help you utilize the full power of fairseq while using it on sequence labeling tasks.



## Example: Training tiny BERT on NER (from scratch) on CoNLL-2003

### 1. Prepare Data

Assumming your data is in the following IOB format: 

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
with the 3 splits `train`, `valid` and `test` in `path/to/data/conll-2003`

Run 
```
python preprocess.py --seqtag-data-dir path/to/data/conll-2003 \
      --destdir path/to/data/conll-2003 \
      --nwordssrc 30000 \
      --bpe sentencepiece \
      --sentencepiece-model /path/to/sentencepiece.bpe.model
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
Training starts:
```
epoch 001 | loss 2.313 | ppl 4.97 | F1-score 0 | wps 202.2 | ups 9.09 | wpb 18 | bsz 1.5 | num_updates 2 | lr 0.005 | gnorm 4.364 | clip 0 | train_wall 0 | wall 0                            
epoch 002 | valid on 'valid' subset | loss 0.557 | ppl 1.47 | F1-score 0.666667 | wps 549.4 | wpb 18 | bsz 1.5 | num_updates 4 | best_F1-score 0.666667                                       
epoch 002:   0%|                                                                                                                                                        | 0/2 [00:00<?, ?it/s]2020-06-05 22:09:03 | INFO | fairseq.checkpoint_utils | saved checkpoint checkpoints/checkpoint_best.pt (epoch 2 @ 4 updates, score 0.6666666666666666) (writing took 0.09897447098046541 seconds)
epoch 002 | loss 1.027 | ppl 2.04 | F1-score 0 | wps 121.8 | ups 6.77 | wpb 18 | bsz 1.5 | num_updates 4 | lr 0.005 | gnorm 2.657 | clip 0 | train_wall 0 | wall 1  
...
```

### 3. Predict and Evaluate
```
python predict.py path/to/data/conll-2003/bin \
         --path checkpoints/checkpoint_last.pt \
         --task sequence_tagging \
         -s source.bpe -t target.bpe \
         --pred-subset test
         --results-path model_outputs/
```
This writes source and prediction to `model_outputs/test.txt` and prints: 
```
    precision    recall  f1-score   support

     PERS     0.7156    0.7506    0.7327       429
      ORG     0.5285    0.5092    0.5187       273
      LOC     0.7275    0.7105    0.7189       342

micro avg     0.6724    0.6743    0.6734      1044
macro avg     0.6706    0.6743    0.6722      1044
```

## TODO

- [x] log F1 metric on validation using Seqeva
- [x] save best model on validation data according to F1 score not loss
- [x] work with BPE
- [x] load and finetune pretrained BERT or RoBERTa 
- [x] prediction/evaluation script
- [ ] LSTM models


