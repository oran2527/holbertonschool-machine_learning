#!/usr/bin/env python3
""" Data set """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Data set """

    def __init__(self):
        """ Data set """
        examples, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                                with_info=True, as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.\
            tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """ Data set """
        tokenizer_pt = tfds.features.text.\
            SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en
                                                  in data),
                                                 target_vocab_size=2 ** 15)
        tokenizer_en = tfds.features.text.\
            SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en
                                                  in data),
                                                 target_vocab_size=2 ** 15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ Data set """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens
