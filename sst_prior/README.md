# Introduction
Prior experiments on SST-2 dataset.

# Requirements
* torch==1.9.1+cu111
* torchvision==0.10.1+cu111
* tensorboardX==2.4
* sklearn==0.0
* transformers==2.9.0

Datasets are available on the GLUE benchmark website (https://gluebenchmark.com/tasks), and pretrained models are available on Huggingface (https://huggingface.co/models).


An example of running prior experiments on SST-2.

1. Preprocess the datasets:
```
python data_process.py --bert_path='roberta-base' --model='RoBERTa'
```

2. Fine-tune a base model with the original fine-tuning approach:
```
python main.py --option='train' --model='RoBERTa' --bert_path='roberta-base'
```

3. Fine-tune a model with different dropping strategies:
```
python main.py --option='train' --do_mask --attribution='GA' --dropping_rate=0.3 --dropping_method='low' --mask_layers='0' --model='RoBERTa' --bert_path='roberta-base'
```
* Set different dropping strategies via the parameters --attribution and --dropping_method

4. Probe a vanilla fine-tuned model on the development set by setting different dropping strategies (as we present in Appendix B):
```
python main.py --option='probe' --dropping_method='low'
```
* The default attribution method is GA, and you can set --dropping_method='high' for dropping from high-attribution positions.