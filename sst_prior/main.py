import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import argparse
import logging
from self_transformers.modeling_roberta import RobertaForSequenceClassification
from self_transformers.modeling_bert import BertForSequenceClassification
from self_transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset import My_dataset
from trainer import train_process, later_analysis

logger = logging.getLogger(__name__)

def set_seed(seed = 0): # set seed

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
    parser = argparse.ArgumentParser(description = 'prior experiments on SST')
    parser.add_argument('--output_dir', type = str, default = 'model', help = 'directory to store model')
    parser.add_argument('--bert_path', type = str, default = 'roberta-base', help = 'path to pre-trained model BERT/RoBERTa')
    parser.add_argument('--layers', type=int, default=12, help='layers of pre-trained model')
    parser.add_argument('--num_class', type = int, default = 2, help = 'number of classes')
    parser.add_argument('--model', type=str, default='RoBERTa', choices=['RoBERTa', 'BERT'], help='choice of the used model')
    parser.add_argument('--epoch', type =  int, default = 10, help = 'epoch of training')
    parser.add_argument('--patient', type=int, default=10, help='patient of early stopping')
    parser.add_argument('--PTM_learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--max_len', type=int, default=100, help='sequence length')

    parser.add_argument('--option', type = str, default = 'train', choices=['train', 'probe'], help = 'train or probe')
    parser.add_argument('--seed', type = int, default = 321, help = 'random seed')
    parser.add_argument('--main_cuda', type = str, default = 'cuda:0', help = 'main GPU')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch size')

    parser.add_argument('--do_mask', action='store_true', help='training with dropping')
    parser.add_argument('--attribution', type=str, default='GA', choices=['AA', 'GA', 'IGA', 'RD'], help='attribution methods')
    parser.add_argument('--dropping_method', type=str, default='low', choices=['low', 'high'], help='dropping from low/high')
    parser.add_argument('--dropping_rate', type=float, default=0.3, help='dropping rate')
    parser.add_argument('--mask_layers', type=str, default='0', help='which layer to perform dropout, and use comma for split')
    parser.add_argument('--model_path', type=str, default='model/finetuned_RoBERTa.pth', help='path of finetuned model to probe')
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
    if  args.model == 'RoBERTa':
        model = RobertaForSequenceClassification.from_pretrained(
            args.bert_path, num_labels=args.num_class, output_attentions=True)
    elif args.model == 'BERT':
        model = BertForSequenceClassification.from_pretrained(
            args.bert_path, num_labels=args.num_class, output_attentions=True)
    else:
        raise NotImplementedError

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.1},
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
        # DataLoader
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
        eval_dataloader = DataLoader(dataset = eval_dataset, batch_size = args.batch_size, shuffle = False)
        # set warm up
        t_total = int(len(train_dataloader) * args.epoch)
        warmup_steps = int(0.06 * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=int(len(train_dataloader) * args.epoch))

        train_process(
            args = args,
            model = model,
            optimizer = optimizer,
            scheduler=scheduler,
            train_dataloader = train_dataloader,
            eval_dataloader = eval_dataloader,
            device = args.device
        )

    elif args.option == 'probe': # conduct prior experiments on fine-tuned model
        '''test_process
        '''

        if args.model == 'RoBERTa':
            model_state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
            model = RobertaForSequenceClassification.from_pretrained(
                args.bert_path, num_labels=args.num_class, output_attentions=True, state_dict=model_state_dict)
        elif args.model == 'BERT':
            model_state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
            model = BertForSequenceClassification.from_pretrained(
            args.bert_path, num_labels=args.num_class, output_attentions=True, state_dict=model_state_dict)
        else:
            raise  NotImplementedError

        eval_dataset = My_dataset(args = args, option = 'val')
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False)

        dropping_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(12):
            print('mask layer:', i)
            args.mask_layer = [i]
            for rate in dropping_rate_list:
                args.dropping_rate = rate
                later_analysis(args, model.to(args.device), eval_dataloader, args.device)

    else:
        raise NotImplementedError
