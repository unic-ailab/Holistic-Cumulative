import os
import nltk
import json
import numpy as np
os.chdir( os.environ['USERPROFILE']+'/downloads/holc')
from configs import Config
config = Config()
import tf_glove

vector_dim = config.dim_word
n_epochs = config.nepochs

def parse_corpus(_x):
    corpus=[]
    for comment in _x:
        corpus.append(nltk.wordpunct_tokenize(" ".join(comment)))
    return corpus


def train_glove_emb(dataset):

    corpus = parse_corpus(dataset)

    model = tf_glove.GloVeModel(embedding_size=vector_dim, context_size=10)

    model.fit_to_corpus(corpus)

    model.train(num_epochs=n_epochs, log_dir='/tmp/tensorflowlogs', summary_batch_interval=1000)

    return np.vstack([np.zeros(vector_dim),model.embeddings])

