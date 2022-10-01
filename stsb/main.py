import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import random
import argparse
import logging
from self_transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel
from self_transformers.modeling_bert import BertForSequenceClassification
from self_transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dataset import My_dataset
from trainer import train_process, test_process

logger = logging.getLogger(__name__)

def set_seed(seed = 0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_opt():
    '''hyper_parameters
    '''
    parser = argparse.ArgumentParser(description = 'Run AD-DROP on STS-B')
    parser.add_argument('--output_dir', type=str, default='model', help='directory to store model') # # roberta-base  / bert-base-uncased
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased', help='path of pre-trained model BERT/RoBERTa')
    parser.add_argument('--layers', type=int, default=12, help='layers of pre-trained model')
    parser.add_argument('--heads', type=int, default=12, help='heads of pre-trained model')
    parser.add_argument('--num_class', type=int, default=1, help='number of classes')
    parser.add_argument('--model', type=str, default='BERT', choices=['RoBERTa', 'BERT'], help='choice of the used model')
    parser.add_argument('--epoch', type =  int, default = 50, help = 'epoch of training')
    parser.add_argument('--patient', type=int, default=5, help='patient of early stopping')
    parser.add_argument('--PTM_learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--max_len', type=int, default=100, help='sequence length')

    parser.add_argument('--option', type = str, default = 'train', choices=['train', 'test'], help = 'train or test')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--main_cuda', type=str, default='cuda:0', help='main GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    parser.add_argument('--do_mask', action='store_true', help='training with dropping or not')
    parser.add_argument('--attribution', type=str, default='GA', choices=['AA', 'GA', 'IGA', 'RD'], help='attribution methods')
    parser.add_argument('--dropping_method', type=str, default='high', choices=['low', 'high'], help='dropping from low/high')
    parser.add_argument('--p_rate', type=float, default=0.3, help='candidate discard region p')
    parser.add_argument('--q_rate', type=float, default=0.2, help='random drop in candidate region')
    parser.add_argument('--mask_layers', type=str, default='0', help='which layers to perform dropout, and use comma for split when applying AD-DROP into muti-layers (e.g., 0,1,2)')
    parser.add_argument('--model_path', type=str, default='model/finetuned_BERT_AD.pth', help='path of the finetuned model to test')
    return parser.parse_args()

if __name__ == '__main__':
    '''
    '''
    args = parse_opt() #hyperparameters
    set_seed(args.seed) #fix-random-seed
    args.mask_layer = [int(layer) for layer in args.mask_layers.split(',')]
    #logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger.info('hyperparameter = {}'.format(args))
    logger.info('this-process-pid = {}'.format(os.getpid()))
    #difine device, model and optimizer
    args.device = torch.device(args.main_cuda if torch.cuda.is_available() else 'cpu')
    # define the chosen model
    if args.model == 'RoBERTa':
        model = RobertaForSequenceClassification.from_pretrained(
            args.bert_path, num_labels=args.num_class, output_attentions=True)
    elif args.model == 'BERT':
        model = BertForSequenceClassification.from_pretrained(
            args.bert_path, num_labels=args.num_class, output_attentions=True)
    else:
        raise NotImplementedError

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.PTM_learning_rate)

    model.to(args.device)
    #running model
    if args.option == 'train':
        '''train_process
        '''
        train_dataset = My_dataset(args = args, option = 'train')
        eval_dataset = My_dataset(args = args, option = 'val')

        logger.info('data_size: train = {}, eval = {},'.format(len(train_dataset), len(eval_dataset)))

        train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
        eval_dataloader = DataLoader(dataset = eval_dataset, batch_size = args.batch_size, shuffle = False)

        train_process(
            args = args,
            model = model,
            optimizer = optimizer,
            # scheduler=scheduler,
            train_dataloader = train_dataloader,
            eval_dataloader = eval_dataloader,
            test_dataloader = eval_dataloader,
            device = args.device
        )

    elif args.option == 'test':
        '''test_process
        '''

        import csv

        # label_map = {0: 'entailment', 1: 'not_entailment'}
        model_state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))

        if args.model == 'RoBERTa':
            model = RobertaForSequenceClassification.from_pretrained(
                args.bert_path, num_labels=args.num_class, output_attentions=True, state_dict=model_state_dict)
        elif args.model == 'BERT':
            model = BertForSequenceClassification.from_pretrained(
                args.bert_path, num_labels=args.num_class, output_attentions=True, state_dict=model_state_dict)
        else:
            raise NotImplementedError

        test_dataset = My_dataset(args=args, option='test')
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

        _, _, preds_id, annos = test_process(args, model, test_dataloader, args.device)

        with open('STS-B.tsv', 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['index', 'prediction'])
            for i, lab in enumerate(preds_id):
                if lab < 0:
                    lab = 0.0
                if lab > 5:
                    lab = 5.0
                tsv_w.writerow([str(i), "%.3f" % lab])
        f.close()
    else:
        raise NotImplementedError
