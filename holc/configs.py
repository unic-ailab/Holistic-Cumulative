import os

class Config():
    ''' set train/test files and pre-set for train/dev/test splits '''
    filename_train    = 'SST.xlsx'
    presetDataset     =  True # preset train/test splits true setting requires text file with splits.
    rootFolder        =  os.environ['USERPROFILE']
    model_env        =   rootFolder + '/downloads/Holistic-Cumulative-main/holc/'
    pathToDatasets    =  '/downloads/Holistic-Cumulative-main/holc/datasets/'
    dataset           =  str(filename_train[0:len(filename_train)-5]).lower()

    ''' set training parameters '''
    train_embeddings  =  False # set true, if model will finetune word embeddings
    pre_trained_embs  =  False # set false, if pre-trained embeddings will be used
    train_w2vec       =  False # set true, if model will train first skip gram vectors
    train_glove       =  False # set true, if model will train first global vectors
    rmv_stop_wrds     =  False # set true, if stop words will be removed at preprocessing
    nepochs           =  35 # the number of epochs to train the model
    dropout           =  0.5 # dropout rate
    batch_size        =  32 # batch size
    lr_method         =  'adam' # set rmsprop or adam or sgd or adagrad for optimization method
    pre_trained_emb   =  'fasttext' # set amazon or fasttext or glove
    lr                =  1.7e-3 # learning rate for optimizer
    l2_regul          =  1e-5 # l2 regurarization for weight regularization
    l2_regul_conv     =  1e-6 # l2 regularization for convolution weights regularization
    n_heads           =  1 # num of parallel processings for attention layer
    n_stacks          =  2 # num of layers for attention module
    dim_word          =  300 # dimension of word embeddings dimension
    kfold_num         =  5 # num of iteration in the k-fold setting
    display_step      =  10 # the num of training iterations to present training info
    test_eval_batch   =  2000 # the number of batch test data to apply in the evaluation loop for preventing memory issues. If enough memory, use a number greater than test size data.
    overfit_threshold =  200 # relates to overfit iterations
    percent_noise     =  0.00 # add noise to data

    ''' inner model parameters '''
    input_size        =  'max' # the input size for data preprocesing
    n_classes         =  5 #  the num of classes to train the model
    n_hidden          =  125 # the num of neurons for blstm, attention and classical layer.

    balancing_factor  =  0.00 # the balancing factor, values range in [0,1] valid steps 0.25
    num_feature_maps  =  128 # the num of feature maps for convolution layers
    filter_sizes      =  [3,2] # the convolution filter sizes
    top_k             =  6 # the num of top-k values to extract from the convolution layers
