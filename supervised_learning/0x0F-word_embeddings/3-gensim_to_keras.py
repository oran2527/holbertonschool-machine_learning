#!/usr/bin/env python3
""" Extract Word2Vec """


def gensim_to_keras(model):
    """ Extract Word2Vec """
    return model.wv.get_keras_embedding(train_embeddings=True)
