import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pickle

class My_dataset(Dataset):
    def __init__(self, args, option):
        self.args = args
        self.option = option
        self.__load_data()
        self.len = len(self.data)

    def __load_data(self):
        if self.args.model == 'RoBERTa':
            with open(os.path.join('data', '{}_roberta.pkl'.format(self.option)), 'rb') as f:
                self.data = pickle.load(f)
        elif self.args.model == 'BERT':
            with open(os.path.join('data', '{}_bert.pkl'.format(self.option)), 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        sample = self.data[index]
        tokens_id = torch.tensor(sample['tokens_id']).long()
        attn_mask = torch.tensor(sample['attention_mask'])
        labels = torch.tensor(sample['label']).float()
        segment_ids = torch.tensor(sample['segment_ids'])
        return tokens_id, attn_mask, segment_ids, labels, torch.tensor(index).long()

    def __len__(self):
        return self.len
