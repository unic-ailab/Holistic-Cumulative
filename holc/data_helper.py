import os
import xlrd
import json
import re
import numpy as np 
import codecs
from nltk import FreqDist
import nltk
import text_preprocess as preprocess
from collections import Counter
import itertools
import csv

big_data_list = ['AMAZON','YELP','AMAZON_lite']

prefixes = "(mr|st|mrs|ms|dr)[.]"

def convert_labels_to_binary(y):
    _y = []
    for label in y:
        if label == 3:
            _y.append(1)
        elif label<3 :
            _y.append(0)
        else:
            _y.append(2)
    return np.array(_y)


def remove_last_stop_char(str):
    _str = str.split(' ')
    if _str[-1] == '.':
        _str.pop(len(_str)-1)
    return ' '.join(c for c in _str )

def replace_num_digit(str):
    _str = str.split(' ')
    _preptoken = []
    for _t in _str:
        if _t.isdigit():
            _t = 'digit'
        _preptoken.append(_t)
    
    return ' '.join(c for c in _preptoken )
    

def split_into_sentences(text):
    text = " " + text + "  "
    # remove unicode
    text = preprocess.removeUnicode(text)
    # replace url address with "url"
    text = preprocess.replaceURL(text)
    # replace "@user" with "atUser"
    text = preprocess.replaceAtUser(text)
    # removes hastag in front of a word
    text = preprocess.removeHashtagInFrontOfWord(text)
    text = preprocess.removeNumbers(text)
    text = preprocess.replaceMultiStopMark(text)
    text = preprocess.removeEmoticons(text)
    # text = tmp
    text = text.replace("\n"," ")
    text = remove_last_stop_char(text)
    text = replace_num_digit(text)
    text = re.sub(prefixes,"\\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    
    text = text.replace(".",".<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    if len(sentences)>1:
        sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def load_csv_data(_filename, n_classes):
    binary = False
    if n_classes==2:
        binary= True
        n_classes = 3
    tmp_x=[];tmp_y=[]
    with open(_filename, 'r',encoding="utf-8", errors='ignore') as f:
        for row in csv.reader(f):
            tmp_x.append(row[0])
            tmp_y.append(row[n_classes-2])
        tmp_x.pop(0)
        tmp_y.pop(0)
        tmp_y = list(map(int, tmp_y))
    if binary:
       idx= find_indices(tmp_y, lambda e: e == 0 or e==2)
       tmp_x = [tmp_x[i] for i in idx]
       tmp_y =[tmp_y[i] for i in idx]
       tmp_y = [1 if y == 2 else 0 for y in tmp_y]
       
    if n_classes==5:
        tmp_y = [t-1 for t in tmp_y]
    
    return tmp_x,tmp_y


def load_data(_filename, n_classes):
    binary = False
    if n_classes==2:
        binary= True
        n_classes = 3
    
    tmp_x=[];tmp_y=[]
    
    workbook = xlrd.open_workbook(_filename)
    worksheet = workbook.sheet_by_index(0)
    # preprocess the data
    for i in range(0, len(worksheet.col(0))):
        tmp=json.dumps(worksheet.cell(i, 0).value)
        tmp = re.sub('"', '', tmp)
        tmp=tmp.lower()
        tmp_x.append(tmp)
        tmp_y.append(int(worksheet.cell(i, n_classes-1).value))
    
    if binary:
       idx= find_indices(tmp_y, lambda e: e == 0 or e==2)
       tmp_x = [tmp_x[i] for i in idx]
       tmp_y =[tmp_y[i] for i in idx]
       tmp_y = [1 if y == 2 else 0 for y in tmp_y]
       
    if n_classes==5:
        tmp_y = [t-1 for t in tmp_y]
    return tmp_x,tmp_y


def maxLength(_x):
    # get the average for each sequence
    list =[]
    for _doc in _x:
       for j in _doc:
           list.append(len(j))
    return np.max(list)//1
        
def maxSentences(_x):
    list = []
    for _doc in _x:
        list.append(len(_doc))
    return np.max(list)//1

def avgLength(_x):
    # get the average for each sequence
    list =[]
    for _doc in _x:
       for j in _doc:
           list.append(len(j))
    return np.average(list)//1
    
def avgSentences(_x):
    list = []
    for _doc in _x:
        list.append(len(_doc))
    return np.average(list)//1

def seq_sent_doc_lengths(x,max_sentences):
    tmp_doc =[]
    for i in range(len(x)):
        tmp_s = x[i]
        tmp_sent =[]
        for j in range(len(tmp_s)):
            tmp_sent.append(len(tmp_s[j]))
        padded_zeros=np.array(np.zeros(max_sentences-len(tmp_sent)))
        tmp_padded_doc = np.concatenate((np.array(tmp_sent) ,padded_zeros),axis=0)
        tmp_doc.append(tmp_padded_doc)
    return tmp_doc
    
def split_load_data_sentences(_x):
    # store document with sentences
    tmp_x = [] 
    # store text only
    _text=''
    for document in range(len(_x)):
        # split document into sentences
        sentences = split_into_sentences(_x[document])
        # store temp sentences
        tmp_sentences = []
        # clear sentences 
        for s in range(len(sentences)):
            # get only the words in the sentence
            tmp=re.sub('\W+',' ', sentences[s]).strip()
            # avoid empty sentences
            if len(tmp)>0:
                # store text only
                _text +=tmp + ' '  
                # store split sentence
                tmp_sentences.append(tmp)
        # store tmp sentences
        tmp_x.append(tmp_sentences)
    return tmp_x

def get_index2word_map(text):
    tokens =  [token for token in text.split() ]
    _map = {i+1:tokens[i] for i in range(len(tokens))}
    return _map

def create_vocabulary_from_domain_opinion_words(domain_opinion_words):
    # create an empty network of model's vocabulary
    def init_vocab_network(n_inputs):
        network = list()
        for i in range(0,n_inputs):
            layer={'value':i,'token':''}
            network.append(layer)
        return network
    
    # initiate model's vocabulary
    _voc=init_vocab_network(len(domain_opinion_words))
    # update vocabulary values
    for index in enumerate(domain_opinion_words):
        _voc[index[0]]['token']=index[1]
    
    return _voc
    
def create_vocabulary(data_x, rmv_stop_wrds):
    # create an empty network of model's vocabulary
    def init_vocab_network(n_inputs):
        network = list()
        for i in range(0,n_inputs):
            layer={'value':i+1,'token':''}
            network.append(layer)
        return network
        
    # given a list of words, return a dictionary of word-frequency pairs.
    def wordlist_to_freq_dict(wrdlist):
        wordfreq = [wrdlist.count(p) for p in wrdlist]
        return dict(zip(wrdlist,wordfreq))

    # sort the dictionary of word-frequency pairs in descending order
    def sort_freq_dict(freqdict):
        aux = [(freqdict[key], key) for key in freqdict]
        aux.sort()
        aux.reverse()
        return aux
    
    def parse_corpus_create_vocab(_x):
        corpus=[]
        for comment in _x:
            corpus.append(nltk.wordpunct_tokenize(" ".join(comment)))
        word_counts = Counter()
        for region in corpus:
            word_counts.update(region)
        return word_counts
    
    
    if rmv_stop_wrds:
        word_freq = parse_corpus_create_vocab(data_x)
        stopwords = nltk.corpus.stopwords.words('english')
        dict_filter = lambda word_freq, stopwords: dict( (word,word_freq[word]) for word in word_freq if word not in stopwords)
        wordlist = dict_filter(word_freq, stopwords)
    else :
        word_freq = parse_corpus_create_vocab(data_x)
        stopwords = []
        dict_filter = lambda word_freq, stopwords: dict( (word,word_freq[word]) for word in word_freq if word not in stopwords)
        wordlist = dict_filter(word_freq, stopwords)
        
    # initiate model's vocabulary
    _voc=init_vocab_network(len(wordlist))
    
    # update vocabulary values
    for index in enumerate(wordlist):
        _voc[index[0]]['token']=index[1]
    return _voc

def dataset_stop_words(tmp_x,vocabulary,_max_seqlen,_max_opinionlen):
    # store data to integer
    _x = []
    _vocab = [token['token'] for token in (vocabulary)]
    for i in range(len(tmp_x)):
        # store tmp sentences
        sentences = tmp_x[i]
        # store tmp data to int sentences
        tmp_sentences = []
        # iterate through tmp sentences 
        for j in range(len(sentences)):
            # get tmp sentence    
            sentence = sentences[j]
            # cut sequence length greater than _max_seqlen value
            if len(sentence.split()) > _max_seqlen:
                tmp_s =''
                for token in sentence.split()[:_max_seqlen]:
                   tmp_s += token + ' '
                # get tmp cut sentence
                sentence = tmp_s.strip().split(' ')
             # map sentence word to integers acording to vocabulary values
            _seq=''
            for token in sentence.split():
                if token in _vocab:
                    _seq += token + ' '
                else :
                    _seq += 'PAD_TOKEN' + ' '
            tmp_sentences.append(_seq.strip())
        _x.append(tmp_sentences)
    return _x

def data_to_integer(tmp_x, _y_labels, vocabulary, _max_seqlen, _max_opinionlen):
    # initial max sequence length
    max_sequence = 0
    def word_to_integer(str,dictionary):
        for index in dictionary:
            tmp_value = 0
            if index['token'] == str:
                tmp_value =index['value']
                break
        return tmp_value  
    
    # store data to integer
    _x_int = []
    
    for i in range(len(tmp_x)):
        # store tmp sentences
        sentences = tmp_x[i]
        # store tmp data to int sentences
        tmp_int_sentences = []
        
        # iterate through tmp sentences 
        for j in range(len(sentences)):
            # cut opinion greater than _max_opinionlen sentenses
            if j >= _max_opinionlen:
                break
            # get tmp sentence    
            sentence = sentences[j]
            # cut sequence length greater than _max_seqlen value
            if len(sentence.split()) > _max_seqlen:
                tmp_s =''
                for sent in sentence.split()[:_max_seqlen]:
                   tmp_s += sent + ' '
                # get tmp cut sentence
                sentence = tmp_s
             # map sentence word to integers acording to vocabulary values
            seq_integer = [word_to_integer(token,vocabulary) for token in sentence.split()]
            
            # update tmp maximum sequence 
            if max_sequence < len(seq_integer):
                max_sequence = len(seq_integer)
            # store converted to integer tmp sentence    
            tmp_int_sentences.append(seq_integer)
        # store converted to integer tmp sentences
        _x_int.append(tmp_int_sentences)
        
    return _x_int,_y_labels, max_sequence

def calculate_document_length(documents):
    return max(len(x) for x in documents)
    
def calculate_sequence_length(num):
    if not num%2==0:
        num+=1
    return num


def pad_sentence_document_mode(documents,_seq_length, padding_word="0"):
    # calculate maximum sentences per opinion 
    document_length = calculate_document_length(documents)
    # calculate maximum sentence length
    sequence_length = calculate_sequence_length(_seq_length)
    
    padded_documents = []
    
    for i in range(len(documents)):
        tmp_padded_document =[]
        tmp_sentences = documents[i]
        if len(tmp_sentences) is 0:
            tmp_sentences = [[0]]
        tmp_padded_sentences=[]
        for j in range(len(tmp_sentences)):
            padded_zeros_words=np.array(np.zeros(sequence_length-len(tmp_sentences[j])))
            tmp_padded_sentence = np.concatenate((padded_zeros_words,np.array(tmp_sentences[j])),axis=0)
            tmp_padded_sentences.append(tmp_padded_sentence)
      
        tmp_padded_document = np.concatenate((np.array(tmp_padded_sentences),
        np.array(np.zeros((document_length-len(tmp_sentences),sequence_length)))),axis=0)
            
        padded_documents.append(tmp_padded_document)
        
    return padded_documents, document_length, sequence_length

def load_word_list(filename):
    word_list = []
    with open(filename,'r') as f:
         list = f.readlines()
         for word in list:
              word = word.rstrip('\n').lower()
              word_list.append(word)
    f.close()
    return word_list

def next_batch(num, data, labels,seqlens,_has_seqns,op_lens):
    
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    seqlens_shufle = [seqlens[ i] for i in idx]
    op_lens_shufle = [op_lens[ i] for i in idx]
    
    if _has_seqns is True:
        return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(seqlens_shufle),np.asarray(op_lens_shufle) 
    else :
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def read_preset_dataset_idxs(filename,n_classes):
    os.chdir(os.environ['USERPROFILE'] +'/downloads/holc/datasets')
    if n_classes==2:
        _filename = (filename[0:len(filename)-5]) + "b.txt"
    else :
        _filename = (filename[0:len(filename)-5]) + ".txt"
    load_file = open(_filename, "r")
    idxs =[]
    for line in load_file:
        idxs.append(line.rstrip('\n').replace('"', ''))
    load_file.close()
    idxs =list(map(int,idxs))
    return idxs
    

def _run_sentence_document_mode_pre_stage(fl_name, rmv_stop_wrds,n_classes,dataset_base,domain_op_words):
    
    print('loading dataset...')
    if(fl_name[0:len(fl_name)-4]in big_data_list):
        os.chdir(os.environ['USERPROFILE'] +'/Google Drive/datasets')
        fl_name= fl_name[0:len(fl_name)-4] + '.csv'
        data_x,data_y = load_csv_data(fl_name,n_classes)  
    else :
        os.chdir(os.environ['USERPROFILE'] +'/downloads/holc/datasets')
        data_x,data_y = load_data(fl_name,n_classes)  
        
    print('calculating ' + dataset_base + ' values...')
    data_x= split_load_data_sentences(data_x)
      
    vocabulary = create_vocabulary(data_x,rmv_stop_wrds)
    
    _ =  " ".join(map(str, [token['token'] for token in vocabulary]))
    _vocab = get_index2word_map(_)
    
    if domain_op_words:
        os.chdir(os.environ['USERPROFILE'] +'/downloads/holc/domain_words')
        domain_op_words_file = open("domain_opinion_aware_words.txt", "r")
        domain_words=[]
        for line in domain_op_words_file:
            domain_words.append(line.rstrip('\n'))
        vocabulary =create_vocabulary_from_domain_opinion_words(domain_words)
        _ =  " ".join(map(str, [token['token'] for token in vocabulary]))
        _vocab = get_index2word_map(_)
    
    
    if rmv_stop_wrds:
        data_x = dataset_stop_words(data_x,vocabulary,1000,1000)
        print('removing stop words...')
        vocabulary = create_vocabulary(data_x,rmv_stop_wrds)
        _ =  " ".join(map(str, [token['token'] for token in vocabulary]))
        _vocab = get_index2word_map(_)
        
    
    vocab_size = len(_vocab)
    print('vocabulary size: %d' % vocab_size)
    
    x, y, _ =  data_to_integer(data_x,data_y,vocabulary,1000,1000)
    
    if dataset_base == 'max':
        # calculate max sentences/document
        _Sentences = int(maxSentences(x))
        # calculate max sequences in the corpus
        _Sequences = int(maxLength(x))
    elif dataset_base=='avg':
        # calculate avg sentences/document
        _Sentences = int(avgSentences(x))
        # calculate max sequences in the corpus
        _Sequences = int(maxLength(x))
        
    return _Sentences, _Sequences,data_x, _vocab, vocab_size,vocabulary

def calculate_opinions_lengths(x):
    op_lengths =[]
    for opinion in x:
        op_lengths.append(len(opinion))
    return np.array(op_lengths)

def create_noise_data(max_seqlens,max_oplens,num_noise_data,vocab_size,n_classes):
    return np.random.randint(vocab_size, size=(num_noise_data,max_oplens,max_seqlens)), np.random.randint(n_classes, size=(num_noise_data)), np.random.randint(max_seqlens+1, size=(num_noise_data,max_oplens)),  np.random.randint(max_oplens+1, size=(num_noise_data))

def add_noise_data_to_train(x,y,data_seqlens,data_oplens,vocab_size,percent_noise,n_classes):
    x_noise,y_noise,data_seqlens_noise,data_oplens_noise = create_noise_data(int(x.shape[2]),int(x.shape[1]),int(percent_noise*len(x)),vocab_size,n_classes)
    x = np.vstack([x,x_noise])
    y = np.concatenate([y,y_noise])
    data_seqlens= np.vstack([data_seqlens,data_seqlens_noise])
    data_oplens =np.concatenate([data_oplens,data_oplens_noise])
    return x,y,data_seqlens,data_oplens
    
def _run_sentence_document_mode(fl_name, max_seqlen, max_opinionlen,rmv_stop_wrds,n_classes,vocabulary,percent_noise):
    # set path folder
    if(fl_name[0:len(fl_name)-4]in big_data_list):
        os.chdir(os.environ['USERPROFILE'] +'/Google Drive/datasets')
        fl_name= fl_name[0:len(fl_name)-4] + '.csv'
        data_x,data_y = load_csv_data(fl_name,n_classes)  
    else :
        os.chdir(os.environ['USERPROFILE'] +'/downloads/holc/datasets')
        data_x,data_y = load_data(fl_name,n_classes)
        
    print('converting to sequences...')
    data_x= split_load_data_sentences(data_x)
    
    print('converting to sequences of integers...')
    x, y, sequence_length =  data_to_integer(data_x,data_y,vocabulary,max_seqlen,max_opinionlen)
    
        
    data_seqlens = seq_sent_doc_lengths(x,max_opinionlen)
    data_oplens = calculate_opinions_lengths(x)
    print('zero padding...')
    x, document_size, max_sequence_length = pad_sentence_document_mode(x,max_seqlen)    
    # convert to numpy
    x =np.array(x,dtype=np.float32)
    y= np.array(y)
    data_seqlens =np.array(data_seqlens)
    data_oplens = np.array(data_oplens)
    
    print('end of preprocessing...')
    
    return x,y,max_sequence_length, document_size,data_seqlens,data_oplens
