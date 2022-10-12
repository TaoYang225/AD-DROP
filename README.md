# AD-DROP: Attribution-Driven Dropout for Robust Language Model Fine-Tuning (NeurIPS 2022)

-----------------------------------------------------
## For GLUE benchmark tasks (SST-2, MNLI, QNLI, QQP, CoLA, STS-B, MRPC, and RTE):
### Environment
* python==3.6.2

### Dependencies
* torch==1.9.1+cu111
* torchvision==0.10.1+cu111
* tensorboardX==2.4
* transformers==2.9.0
* sklearn

Datasets are available on the GLUE benchmark website (https://gluebenchmark.com/tasks), and pre-trained models are available on Huggingface (https://huggingface.co/models).
In the current version, the supported pre-trained models are BERT/RoBERTa series models.

An example of fine-tuning BERT with AD-DROP.

1. Preprocess the datasets:
```
python data_process.py --bert_path='bert-base-uncased' --model='BERT'
```

2. (Optional) Fine-tune a base model with the original fine-tuning approach:
```
python main.py --option='train' --model='BERT' --bert_path='bert-base-uncased'
```
  or  
```bash
>> bash run_ft.sh
```

3. Fine-tune a model with our AD-DROP:
```
python main.py --option='train' --do_mask --attribution='GA' --p_rate=0.3 --q_rate=0.3 --mask_layers='0' --moedl='BERT' --bert_path='bert-base-uncased'
```
* Set different dropping strategies via the parameter --attribution (options  including ['GA', 'AA', 'IGA', 'RD'].).
* Set the parameter --mask_layers='0,1,2,3' to apply AD-DROP in multiple layers.

- We provide a script for searching for the best settings of 'p_rate' and 'q_rate':
```bash
>> bash run_addrop.sh
```
* The script will save all log files automatically. We provide 'log2excel.py' to collect the best settings.

4. Test the fine-tuned model on the test set:
```
python main.py --option='test' --main_cuda='cpu' --model_path='model/finetuned_BERT_AD.pth'
```
* It will generate a '.tsv' file for evaluation on the GLUE leaderboard.

-----------------------------------------------------
## For token-level tasks (NER and Translation):
### Environment
* python==3.6.2

### Dependencies
* torch==1.9.1+cu111
* **transformers==4.7.0**
* sacrebleu==2.2.0

The supported pre-trained models are ELECTRA and OPUS-MT. 
We perform the two tasks by following the official colab. Please refer to HuggingFace [Token Classification](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb) and [Translation](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation.ipynb) for details.

-----------------------------------------------------

## For cite the paper:
```bibtex
@inproceedings{
yang2022addrop,
title={{AD}-{DROP}: Attribution-Driven Dropout for Robust Language Model Fine-Tuning},
author={Tao Yang and Jinghao Deng and Xiaojun Quan and Qifan Wang and Shaoliang Nie},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022}
}
```
