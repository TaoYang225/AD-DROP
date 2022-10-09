# AD-DROP: Attribution Driven Dropout for Robust Language Model Finetuning (NeurIPS 2022)

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

Datasets are available in the GLUE benchmark (https://gluebenchmark.com/tasks), and pre-trained models are available in Huggingface (https://huggingface.co/models).
You should download the dataset and put it into the corresponding folder. In the current version, the supported pretrained models are BERT/RoBERTa series models.

An example of finetuning BERT with AD-DROP.

1. Preprocess the dataset:
```
python data_process.py --bert_path='bert-base-uncased' --model='BERT'
```

2. (Optional) Vanilla finetune a basic model:
```
python main.py --option='train' --model='BERT' --bert_path='bert-base-uncased'
```
  or  
```bash
>> bash run_ft.sh
```

3. Finetune a model with AD-DROP:
```
python main.py --option='train' --do_mask --attribution='GA' --p_rate=0.3 --q_rate=0.3 --mask_layers='0' --moedl='BERT' --bert_path='bert-base-uncased'
```
* You can set different dropping strategies via the parameter --attribution (it can choose from ['GA', 'AA', 'IGA', 'RD'].).
* You can set the parameter --mask_layers='0,1,2,3' to apply AD-DROP into multi-layers.

- We provided a script for searching the best settings of 'p_rate' and 'q_rate':
```bash
>> bash run_addrop.sh
```
* The script will save all log files automatically. We provided 'log2excel.py' to extract the best settings.

4. Test the finetuned model on the test set:
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

The supported pretrained models are ELECTRA and OPUS-MT. 
We implement the two tasks by following the official colab. Please refer to HuggingFace [Token Classification](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb) and [Translation](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation.ipynb) for details.

-----------------------------------------------------
