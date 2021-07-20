#!/usr/bin/env python3
""" Semantic Search """
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """ Semantic Search """
    # Embedding model from tf-hub
    embed = hub.load("https://tfhub.dev/google/"
                     "universal-sentence-encoder-large/5")
    # Read file and put it content into a list
    references = [sentence]
    for file in os.listdir("./{}".format(corpus_path)):
        if file.endswith(".md"):
            with open('./{}/{}'.format(corpus_path, file)) as f:
                references.append(f.read())
    # Embedding to the content of the files
    embeddings = embed(references)
    # Create a correlation matrix
    correlation = np.inner(embeddings, embeddings)
    # Best option between the sentence and all references
    closest = np.argmax(correlation[0, 1:])
    return references[closest + 1]
