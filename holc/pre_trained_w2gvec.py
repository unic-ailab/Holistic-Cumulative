import zipfile
import numpy as np
import logging
import json
import os

''' https://nlp.stanford.edu/projects/glove/ '''
def glovepath(size):
    if size==300:
        if(os.path.isfile(os.environ['USERPROFILE'] + "/downloads/Holistic-Cumulative-main/holc/embeddings/glove.840B.300d.zip")):
            path_to_glove = os.environ['USERPROFILE'] + "/downloads/Holistic-Cumulative-main/holc/embeddings/glove.840B.300d.zip"
            glovefile = 'glove.840B.300d.txt'
            GLOVE_SIZE = 300
    elif size == 200:
        if(os.path.isfile(os.environ['USERPROFILE'] + "/downloads/Holistic-Cumulative-main/holc/embeddings/glove.6B.zip")):
            path_to_glove = os.environ['USERPROFILE'] + "/downloads/Holistic-Cumulative-main/holc/embeddings/glove.6B.zip"
            glovefile = 'glove.6B.200d.txt'
            GLOVE_SIZE = 200
    elif size ==100:
        if(os.path.isfile(os.environ['USERPROFILE'] + "/downloads/Holistic-Cumulative-main/holc/embeddings/glove.6B.zip")):
            path_to_glove = os.environ['USERPROFILE'] + "/downloads/Holistic-Cumulative-main/holc/embeddings/glove.6B.zip"
            glovefile = 'glove.6B.100d.txt'
            GLOVE_SIZE = 100
    return path_to_glove,glovefile,GLOVE_SIZE


def get_index2word_map(text):
    tokens = word_tokenize(text)
    vocab = sorted(set(tokens))
    _map = {i+1:vocab[i] for i in range(len(vocab))}
    return _map

def get_glove(path_to_glove, word2index_map,glovefile):
    embedding_weights = {}
    found_tokens =[]
    count_all_words = 0
    total_words =len(word2index_map)
    with zipfile.ZipFile(path_to_glove) as z:
        with z.open(glovefile) as f:
            for line in f:
                vals = line.split()
                word = str(vals[0].decode("utf-8"))
                if word in word2index_map:
                    found_tokens.append(word)
                    count_all_words += 1
                    if count_all_words % 100 == 0:
                        print("pre-trained token " + str(count_all_words) + ' from ' + str(total_words))
                    coefs = np.asarray(vals[1:], dtype='float32')
                    coefs /= np.linalg.norm(coefs)
                    embedding_weights[word] = coefs
                if count_all_words == len(word2index_map) - 1:
                    break
        f.close
        z.close
        print("pre-trained token " + str(count_all_words) + ' from ' + str(total_words))
    return embedding_weights,found_tokens

def get_not_found_tokens(_found_tokens,index2word_map):
    not_found_tokens=[]
    for index, token in index2word_map.items():
        if token not in _found_tokens:
            not_found_tokens.append(token)
    return not_found_tokens

def pad_not_found_tokens(_embeddings_dict, not_found_tokens,GLOVE_SIZE):
    for token in not_found_tokens:
        _embeddings_dict[token]= np.random.uniform(-.2,.2,GLOVE_SIZE)
    return _embeddings_dict

def getPretrainedWordVextors(index2word_map,emb_size):
    path_to_glove,glovefile,GLOVE_SIZE = glovepath(emb_size)
    index2word_map[0] = "pad_token"
    word2index_map = {word: index for index, word in index2word_map.items()}
    vocabulary_size = len(index2word_map)
    word2embedding_dict, found_tokens = get_glove(path_to_glove, word2index_map,glovefile)
    not_found_tokens= get_not_found_tokens(found_tokens,index2word_map)
    word2embedding_dict = pad_not_found_tokens(word2embedding_dict,not_found_tokens,GLOVE_SIZE)
    embedding_matrix = np.zeros((vocabulary_size, GLOVE_SIZE),dtype=np.float32)

    for word, index in word2index_map.items():
        if not word == "pad_token":
            word_embedding = word2embedding_dict[word]
            embedding_matrix[index, :] = word_embedding

    return embedding_matrix




