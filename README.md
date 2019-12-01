# Textsum for chinese dialogues
# 生成式中文文本摘要模型
## Use seq2seq with Bahdanau attention, tensorflow 2.0
## Implementation:
* fasttext embedding
* attention coverage loss
* beam search

* bucket batch, longer/varing sequence: low traing performance can be solved by adding input_signature, and when max length is very long, OOM happens often, which can be solved by decreasing batch_size, but this can slow down the training speed actually.

## TODO:
* scheduled sampling
* embedding finetuning
* tf-idf based / sequence length based loss
* implementation in paddlepaddle
* Pgen