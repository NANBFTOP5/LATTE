import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class MedMent():
    def __init__(self, args):
        path = 'data/test_path'
        dataset_path = path + '/Medmentions/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'

        print("preprocessing data files...")
        if not os.path.exists('{}/{}l'.format(path, args.train_file)):
            self.preprocess_file('{}/{}'.format(path, args.train_file))
        if not os.path.exists('{}/{}l'.format(path, args.dev_file)):
            self.preprocess_file('{}/{}'.format(path, args.dev_file))

        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       'p_label': ('p_label', self.LABEL),
                       'n_label': ('n_label', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'positive': [('p_word', self.WORD), ('p_char', self.CHAR)],
                       'negative': [('n_word', self.WORD), ('n_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('p_label', self.LABEL), ('n_label', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('p_word', self.WORD), ('p_char', self.CHAR),
                       ('n_word', self.WORD), ('n_char', self.CHAR)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.dev = data.TabularDataset.splits(
                path=path,
                train='{}l'.format(args.train_file),
                validation='{}l'.format(args.dev_file),
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        #cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))

        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.train_iter = data.BucketIterator(
            self.train,
            batch_size=args.train_batch_size,
            device=device,
            repeat=True,
            shuffle=True,
            sort_key=lambda x: len(x.c_word)
        )

        self.dev_iter = data.BucketIterator(
            self.dev,
            batch_size=args.dev_batch_size,
            device=device,
            repeat=False,
            sort_key=lambda x: len(x.c_word)
        )

        # self.train_iter, self.dev_iter = \
        #    data.BucketIterator.splits((self.train, self.dev),
        #                               batch_sizes=[args.train_batch_size, args.dev_batch_size],
        #                               device=device,
        #                               sort_key=lambda x: len(x.c_word))
