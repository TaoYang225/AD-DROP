# Introduction
The source code of prior experiments on SST-2 dataset, and this document describes how to run the code.

# Requirements
* torch==1.9.1+cu111
* torchvision==0.10.1+cu111
* tensorboardX==2.4
* sklearn==0.0
* transformers==2.9.0

Datasets are available in the GLUE benchmark (https://gluebenchmark.com/tasks), and pre-trained models are available in Huggingface (https://huggingface.co/models).
You should download the dataset and set the pre-trained model ('bert-base-uncased', 'roberta-base' or larger models).


An example for running prior experiments on SST-2.

1. To process the dataset, run:
```
python data_process.py --bert_path='roberta-base' --model='RoBERTa'
```

2. To original finetune a basic model, run:
```
python main.py --option='train' --model='RoBERTa' --bert_path='roberta-base'
```

3. To finetune a model with different dropping strategies, run:
```
python main.py --option='train' --do_mask --attribution='GA' --dropping_rate=0.3 --dropping_method='low' --mask_layers='0' --model='RoBERTa' --bert_path='roberta-base'
```
* You can set different dropping strategies via the parameters --attribution and --dropping_method

4. To probe a vanilla finetuned model on the development set by setting different dropping strategies as we present in Appendix B, run:
```
python main.py --option='probe' --dropping_method='low'
```
* Here, the default attribution method is GA, and we also can set --dropping_method='high' for dropping high attribution positions.