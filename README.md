# Textsum for chinese dialogues
# 生成式中文文本摘要模型
## Use seq2seq with Bahdanau attention, tensorflow 2.0
## Implementation:
* fasttext embedding
* attention coverage loss
* beam search

* bucket batch, longer/varing sequence: low traing performance solved by adding input_signature

## TODO:
* scheduled sampling
* embedding fintuning
* tf-idf based / sequence length based loss
* implementation in paddlepaddle
* Pgen