#!/usr/bin/env python3
""" Data set """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Data set """
    def __init__(self, batch_size, max_len):
        """ Data set """
        def filter_max_length(x, y, max_length=max_len):
            """ Filter function """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True, as_supervised=True)
        train = examples['train']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(train)
        data_train = train.map(self.tf_encode)
        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        train_size = metadata.splits['train'].num_examples
        data_train = data_train.shuffle(train_size).\
            padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
        data_valid = examples['validation'].map(self.tf_encode)
        self.data_valid = data_valid.filter(filter_max_length).\
            padded_batch(batch_size, padded_shapes=([None], [None]))

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

    def tf_encode(self, pt, en):
        """ Data set """
        result_pt, result_en = tf.py_function(self.encode, [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
