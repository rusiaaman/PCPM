# PCPM

**P**resenting **C**orpus of **P**retrained **M**odels. Links to pretrained models in NLP and voice with training script. 

With rapid progress in NLP it is becoming easier to bootstrap a machine learning project involving text. Instead of starting with a base code, one can now start with a base pretrained model and within a few iterations get SOTA performance. This repository is made with the view that pretrained models minimizes collective human effort and cost of resources, thus accelerating development in the field.

Models listed are curated for either pytorch or tensorflow because of their wide usage. 

Note: [`pytorch-transofmers`](https://github.com/huggingface/pytorch-transformers) is an awesome library which can be used to quickly infer/fine-tune from many pre-trained models in NLP. The pre-trained models from those are not included here.

## Contents

* [Text ML models](#text-ml)
* [Speech to text models](#speech-to-text)
* [Datasets](#datasets)
* [Hall of Shame](#hall-of-shame)
* [Non english models](#non-english)
* [Other Collections](#other-collections)


# Text ML

## Language Models

Name     |      Link      |  Trained On | Training script|
-------|----------|:--------------:|------------:|
Transformer-xl |  https://github.com/kimiyoung/transformer-xl/tree/master/tf#obtain-and-evaluate-pretrained-sota-models | [`enwik8`](#enwik8), [`lm1b`](#lm1b), [`wt103`](#wt103), [`text8`](#text8) |  https://github.com/kimiyoung/transformer-xl |
GPT-2 | https://github.com/openai/gpt-2/blob/master/download_model.py | [`webtext`](#webtext) | https://github.com/nshepperd/gpt-2/ |
Adaptive Inputs (fairseq) | https://github.com/pytorch/fairseq/blob/master/examples/language_model/README.md#pre-trained-models | [`lm1b`](#lm1b) | https://github.com/pytorch/fairseq/blob/master/examples/language_model/README.md

## Permutation lanugage modelling Based - XLNet

Name     |      Link      |  Trained On | Training script|
-------|----------|:--------------:|------------:|
XLnet | https://github.com/zihangdai/xlnet/#released-models | [`booksCorpus`](booksCorpus)+[`English Wikipedia`](english-wikipedia)+[`Giga5`](https://catalog.ldc.upenn.edu/LDC2011T07)+[`ClueWeb 2012-B`](https://lemurproject.org/clueweb12/)+[`Common Crawl`](#common-crawl) | https://github.com/zihangdai/xlnet/

## Masked Language Modelling Based - Bert
Name     |      Link      |  Trained On | Training script|
-------|----------|:--------------:|------------:|
RoBERTa | https://github.com/pytorch/fairseq/tree/master/examples/roberta#pre-trained-models | [booksCorpus](booksCorpus)+CC-N EWS+[OpenWebText](openwebtext)+CommonCrawl-Stories |https://github.com/huggingface/transformers |
BERT | https://github.com/google-research/bert/ | [booksCorpus](booksCorpus)+[English Wikipedia](english-wikipedia) | https://github.com/huggingface/transformers|
MT-DNN |   https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt (https://github.com/namisan/mt-dnn/blob/master/download.sh)| [glue](glue)  | https://github.com/namisan/mt-dnn |

## Machine Translation
Name     |      Link      |  Trained On | Training script|
-------|----------|:--------------:|------------:|
OpenNMT | http://opennmt.net/Models-py/ (pytorch) http://opennmt.net/Models-tf/ (tensorflow) | English-German | https://github.com/OpenNMT/OpenNMT-py |
Fairseq (multiple models) | https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md#pre-trained-models | WMT14 English-French, WMT16 English-German | https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md

## Sentiment
Name     |      Link      |  Trained On | Training script|
-------|----------|:--------------:|------------:|
Nvidia sentiment-discovery | https://github.com/NVIDIA/sentiment-discovery#pretrained-models | [SST](#sst), [imdb](#IMDB), [Semeval-2018-tweet-emotion](#Semeval2018te) | https://github.com/NVIDIA/sentiment-discovery |
MT-DNN Sentiment | https://drive.google.com/open?id=1-ld8_WpdQVDjPeYhb3AK8XYLGlZEbs-l | [SST](#sst) | https://github.com/namisan/mt-dnn |


## Reading Comprehension
### SQUAD 1.1
Rank | Name     |      Link      | Training script|
-------|-------|----------|:--------------:|
49 | BiDaf | https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz | https://github.com/allenai/allennlp | 

## Summarization
Model for English summarization

Name     |      Link      |  Trained On | Training script|
-------|-------|----------|:--------------:|
OpenNMT | http://opennmt.net/Models-py/ | Gigaword standard | https://github.com/OpenNMT/OpenNMT-py |


# Speech to Text
Name     |      Link      |  Trained On | Training script | 
-------|----------|:--------------:|------------:|
OpenSeq2Seq-Jasper | https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition.html#models | [librispeech](#librispeech) | https://github.com/NVIDIA/OpenSeq2Seq
Espnet | https://github.com/espnet/espnet#asr-results | [librispeech](#librispeech),Aishell,HKUST,TEDLIUM2 | https://github.com/espnet/espnet
wav2letter++ | https://talonvoice.com/research/ | [librispeech](#librispeech) | https://github.com/facebookresearch/wav2letter
Deepspeech2 pytorch | https://github.com/SeanNaren/deepspeech.pytorch/issues/299#issuecomment-394658265 | [librispeech](#librispeech) | https://github.com/SeanNaren/deepspeech.pytorch
Deepspeech | https://github.com/mozilla/DeepSpeech#getting-the-pre-trained-model | [mozilla-common-voice](#mozilla-common-voice), [librispeech](#librispeech), [fisher](#fisher), [switchboard](#switchboard) | https://github.com/mozilla/DeepSpeech
speech-to-text-wavenet | https://github.com/buriburisuri/speech-to-text-wavenet#pre-trained-models | [vctk](#vctk) | https://github.com/buriburisuri/speech-to-text-wavenet
at16k | https://github.com/at16k/at16k#download-models | NA | NA

# Datasets
Datasets referenced in this document

## Language Model data
### Common crawl
http://commoncrawl.org/
### enwik8
Wikipedia data dump (Large text compression benchmark)
http://mattmahoney.net/dc/textdata.html

### text8
Wikipedia cleaned text (Large text compression benchmark)
http://mattmahoney.net/dc/textdata.html

### lm1b
1 Billion Word Language Model Benchmark
https://www.statmt.org/lm-benchmark/

### wt103
Wikitext 103 
https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

### webtext
Original dataset not released by the authors. An open source collection is available at https://skylion007.github.io/OpenWebTextCorpus/

### English wikipedia
https://en.wikipedia.org/wiki/Wikipedia:Database_download#English-language_Wikipedia

### BooksCorpus
https://yknzhu.wixsite.com/mbweb
https://github.com/soskek/bookcorpus

## Sentiment
### SST
Stanford sentiment tree bank https://nlp.stanford.edu/sentiment/index.html. One of the [Glue](#glue) tasks.

### IMDB
IMDB movie review dataset used for sentiment classification http://ai.stanford.edu/~amaas/data/sentiment

### Semeval2018te
Semeval 2018 tweet emotion dataset https://competitions.codalab.org/competitions/17751

## Glue

Glue is a collection of resources for benchmarking natural language systems. https://gluebenchmark.com/ Contains datasets on natural language inference, sentiment classification, paraphrase detection, similarity matching and lingusitc acceptability. 

## Speech to text data
### fisher
https://pdfs.semanticscholar.org/a723/97679079439b075de815553c7b687ccfa886.pdf

### librispeech
www.danielpovey.com/files/2015_icassp_librispeech.pdf

### switchboard
https://ieeexplore.ieee.org/document/225858/

### Mozilla common voice
https://github.com/mozilla/voice-web

### vctk
https://datashare.is.ed.ac.uk/handle/10283/2651

# Hall of Shame

High quality research which doesn't include pretrained models and/or code for public use.

- **KERMIT** https://arxiv.org/abs/1906.01604 
   Generative Insertion-Based Modeling for Sequences. No code.

# Non English

# Other Collections

## Allen NLP
Built on pytorch, allen nlp has produced SOTA models and open sourced them.
https://github.com/allenai/allennlp/blob/master/MODELS.md

They have neat interactive demo on various tasks at https://demo.allennlp.org/

## GluonNLP
Based on MXNet this library has extensive list of pretrained models on various tasks in NLP.
http://gluon-nlp.mxnet.io/master/index.html#model-zoo


