# PCPM

**P**retrained **C**ollection of **P**retrained **M**odels. Links to pretrained models in NLP and voice with training script. 

With rapid progress in NLP it is becoming easier to bootstrap a machine learning project involving text. Instead of starting with a base code, one can now start with a base pretrained model and within a few iterations get SOTA performance. This repository is made with the view that pretrained models minimizes collective human effort and cost of resources, thus accelerating development in the field.

Models listed are curated for either pytorch or tensorflow because of their wide usage. 

## Contents

* [Text ML models](#text-ml)
* [Voice ML models](#voice-ml)
* [Datasets](#datasets)
* [Hall of Shame](#hall-of-shame)
* [Non english models](#non-english)


# Text ML

**Language Models**

Name     |      Link      |  Trained On | Training script|
-------|----------|:--------------:|------------:|
Transformer-xl |  https://github.com/kimiyoung/transformer-xl/tree/master/tf#obtain-and-evaluate-pretrained-sota-models | [`enwik8`](#enwik8), [`lm1b`](#lm1b), [`wt103`](#wt103), [`text8`](#text8) |  https://github.com/kimiyoung/transformer-xl |
GPT-2 | https://github.com/openai/gpt-2/blob/master/download_model.py | [`webtext`](#webtext) | https://github.com/nshepperd/gpt-2/ |



# Voice ML
Name     |      Link      |  Trained On | Training script | 
-------|----------|:--------------:|------------:|
Deepspeech | https://github.com/mozilla/DeepSpeech#getting-the-pre-trained-model | [librispeech](#librispeech), [fisher](#fisher), [switchboard](#switchboard) | https://github.com/mozilla/DeepSpeech

# Datasets
Datasets referenced in this document

## Language Model data
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
https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

### webtext
Original dataset not released by the authors. An open source collection is available at https://skylion007.github.io/OpenWebTextCorpus/

## Speech to text data
### fisher
https://pdfs.semanticscholar.org/a723/97679079439b075de815553c7b687ccfa886.pdf

### librispeech
www.danielpovey.com/files/2015_icassp_librispeech.pdf

### switchboard
https://ieeexplore.ieee.org/document/225858/

# Hall of Shame

High quality research which doesn't include pretrained models for public use.

# Non English
