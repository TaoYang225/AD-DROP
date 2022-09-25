import pickle
import csv
import pandas as pd
import argparse
from self_transformers.tokenization_roberta import RobertaTokenizer
from self_transformers.tokenization_bert import BertTokenizer

def parse_opt():
    '''hyper_parameters
    '''
    parser = argparse.ArgumentParser(description = 'data process')
    parser.add_argument('--bert_path', type = str, default = 'roberta-base', help = 'path to pre-trained model BERT/RoBERTa')
    parser.add_argument('--model', type=str, default='RoBERTa', help='choice of the used model')

    parser.add_argument('--dataset', type=str, default='sst', help='set dataset')
    parser.add_argument('--train_data_path', type=str, default='data/SST-2/train.tsv', help='set training set path')
    parser.add_argument('--dev_data_path', type=str, default='data/SST-2/dev.tsv', help='set development set path')
    parser.add_argument('--test_data_path', type=str, default='data/SST-2/test.tsv', help='set test set path')
    parser.add_argument('--max_len', type = int, default = 100, help = 'sequence length')

    return parser.parse_args()

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def create_examples(lines, set_type, is_test=False):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, line[0])
        if i == 0:
            continue
        if is_test:
            text_a = line[0]
            examples.append({'guid': guid, 'text_a': text_a})
        else:
            text_a = line[0]
            label = line[1]
            examples.append({'guid':guid, 'text_a':text_a, 'label':label})
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tokenizer_text(args, texts, max_len, tokenizer, is_test=False):
    data = []
    label_map={'0':0, '1':1}
    length = []
    for i,text in enumerate(texts):
        # print(i,text)
        tokens_a = tokenizer.tokenize(text['text_a'].strip())
        length.append(len(tokens_a))
        tokens, input_mask = pad_input_tokens(args, tokens_a, max_len)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        # assert len(segment_ids) == max_len
        #
        if is_test:
            label_id = 0
        else:
            label_id = label_map[text['label']]
        #
        data.append({'tokens_id': input_ids, 'attention_mask':input_mask, 'label':int(label_id)})

    return data

def pad_input_tokens(args, tokens, max_len):

    if args.model == 'RoBERTa':
        context = ["<s>"] + tokens[:max_len - 2] + ["</s>"]
    elif args.model == 'BERT':
        context = ["[CLS]"] + tokens[:max_len - 2] + ["[SEP]"]
    else:
        raise NotImplementedError
    attention_mask = [1] * len(context)

    while len(context) < max_len:
        if args.model == 'RoBERTa':
            context.append("<pad>")
        elif args.model == 'BERT':
            context.append("[PAD]")
        else:
            raise NotImplementedError
        attention_mask.append(0)

    assert len(context) == len(attention_mask)

    return context, attention_mask

if __name__ == '__main__':
    # hyperparameters
    args = parse_opt()
    if args.model == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_path)  # define tokenizer
    elif args.model == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)  # define tokenizer
    else:
        raise  NotImplementedError

    train_texts = read_tsv(args.train_data_path)
    dev_texts = read_tsv(args.dev_data_path)
    test_texts = read_tsv(args.test_data_path)

    train_examples = create_examples(train_texts, "train")
    dev_examples = create_examples(dev_texts, "dev")
    test_examples = create_examples(test_texts, "test", is_test=True)

    train_data = tokenizer_text(args, train_examples, args.max_len, tokenizer)
    dev_data = tokenizer_text(args, dev_examples, args.max_len, tokenizer)
    test_data = tokenizer_text(args, test_examples, args.max_len, tokenizer, is_test=True)

    if args.model == 'RoBERTa':
        with open('data/train_roberta.pkl', 'wb') as wt:
            pickle.dump(train_data, wt)

        with open('data/val_roberta.pkl', 'wb') as wt:
            pickle.dump(dev_data, wt)

        with open('data/test_roberta.pkl', 'wb') as wt:
            pickle.dump(test_data, wt)

    elif args.model == 'BERT':
        with open('data/train_bert.pkl', 'wb') as wt:
            pickle.dump(train_data, wt)

        with open('data/val_bert.pkl', 'wb') as wt:
            pickle.dump(dev_data, wt)

        with open('data/test_bert.pkl', 'wb') as wt:
            pickle.dump(test_data, wt)

    else:
        raise NotImplementedError






