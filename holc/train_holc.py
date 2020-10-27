import tensorflow as tf
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

''' local folder to load scripts and libraries ''' 
os.chdir(os.environ['USERPROFILE'] +'/downloads/holc')
from configs import Config
from holc_model import holcModel
import data_helper
import pre_trained_w2avec as w2avec 
import pre_trained_w2gvec as w2gvec 
import pre_trained_w2fvec as w2fvec 
from learn_metrics import calcMetric
import word2vec
import word2gvec
''' load, set model parameters '''
config = Config()
_config = tf.ConfigProto() 
dyn_dropout, _ = calcMetric.calcDropout(0,0,config.dropout,0)

print('train file:', (config.filename_train[0:len(config.filename_train)-5]))

'''' preprocess the dataset, calculate avg/max input size values, and sentence length '''
_oplen,_seqlen,_sentences,_vocab,vocab_size,_vocab_R=data_helper._run_sentence_document_mode_pre_stage(config.filename_train, config.rmv_stop_wrds,config.n_classes,config.input_size,False)

print(config.input_size + ' Sentences/Sequences: ' + str(_oplen) + ' / ' + str(_seqlen))

x_,y_,sentence_size,opinion_size,seqlengths,op_lengths =  data_helper._run_sentence_document_mode(config.filename_train,_seqlen,_oplen,config.rmv_stop_wrds,config.n_classes,_vocab_R,config.percent_noise) 

'''' convert dataset to numpy '''
x_ = np.array(x_,dtype=np.float32)
y_ = np.eye(int(np.max(y_) + 1))[np.int32(y_)]
seqlengths = np.array(seqlengths,dtype=np.int32)
op_lengths = np.array(op_lengths,dtype=np.int32)

'''' train or load pre-trained word embeddings '''
if config.train_w2vec:
    print('training word embeddings SKIP-GRAM/NCE(8)...')
    embeddings= word2vec.trainWordVectors(_sentences)
elif config.pre_trained_embs:
    if config.pre_trained_emb == 'amazon':
        print('Amazon pre-trained embeddings')
        print('-----------------------------')
        embeddings = w2avec.getPretrainedWordVextors(_vocab)
    elif config.pre_trained_emb == 'glove':
        print('Glove pre-trained embeddings')
        print('----------------------------')
        embeddings = w2gvec.getPretrainedWordVextors(_vocab,config.dim_word)
    elif config.pre_trained_emb == 'fasttext':
        print('Fasttext pre-trained embeddings')
        print('-------------------------------')
        embeddings = w2fvec.getPretrainedWordVextors(_vocab)
elif config.train_glove:
    print('collecting word embeddings GLOVE(' + str(config.dim_word) +')...')
    embeddings = word2gvec.train_glove_emb(_sentences)
else :
    embeddings=np.zeros([vocab_size,config.dim_word])

'''' monitor test accuracy for every loop '''
metric_list = []
_kfold = 0
# prepare k-fold validation
kfold = KFold(config.kfold_num, shuffle=False)
# enumerate splits
for train_idx,test_idx in kfold.split(x_):
    print('creating train/test datasets...')
    if config.presetDataset:
        ids_train,ids_dev,ids_test = data_helper.read_preset_dataset_idxs(config.filename_train,config.n_classes)
        # train dataset
        x_train = x_[:ids_train+ids_dev]
        y_train= y_[:ids_train+ids_dev]
        seqlen_train = seqlengths[:ids_train+ids_dev]
        op_lengths_train= op_lengths[:ids_train+ids_dev]
        x_test = x_[-ids_test:]
        y_test = y_[-ids_test:]
        seqlen_test=seqlengths[-ids_test:]
        op_lengths_test = op_lengths[-ids_test:]
        x_train, x_dev,y_train,y_dev,seqlen_train,seqlen_dev,op_lengths_train,op_lengths_dev, =train_test_split(x_train,y_train,seqlen_train,op_lengths_train,test_size=0.1)
    elif not config.presetDataset: # k-fold validation 
        x_train = x_[train_idx]
        y_train = y_[train_idx]
        seqlen_train = seqlengths[train_idx]
        op_lengths_train= op_lengths[train_idx]
        x_test = x_[test_idx]
        y_test = y_[test_idx]
        seqlen_test=seqlengths[test_idx]
        op_lengths_test = op_lengths[test_idx]
        x_train, x_dev,y_train,y_dev,seqlen_train,seqlen_dev,op_lengths_train,op_lengths_dev, =train_test_split(x_train,y_train,seqlen_train,op_lengths_train,test_size=0.1)
    
    print('dataset: ' + str(len(x_))  + ' train/dev/test ' + str(len(x_train)) + '/' + str(len(x_dev)) +'/' + str(len(x_test)))
    
    # calculate training iterations
    training_iters = int(config.nepochs * (int(len(x_train))/config.batch_size))
    
    print()
    print('Model Parameters')
    print('-------------------')
    print('training classes: ' + str(config.n_classes))
    print('n_hidden: ' + str(config.n_hidden))
    print('embedding_size: ' + str(config.dim_word))
    print('train_embeddings: ' + str(config.train_embeddings))
    print('base_dropout: ' + str(config.dropout))
    print('n_stacks: ' + str(config.n_stacks))
    print('n_heads: ' + str(config.n_heads))
    print('n_feature_maps: ' + str(config.num_feature_maps))
    print('top_k: ' + str(config.top_k))
    print('filter_sizes: ' + str(config.filter_sizes))
    print('balancing_factor: ' + str(config.balancing_factor))
    print('n_epochs: ' + str(config.nepochs))
    print('learning method: ' + str(config.lr_method))
    print('learning rate: ' + str(config.lr))
    print('batch_size: ' + str(config.batch_size))
    print('-------------------')
    print()
    print('training iterations: ' + str(training_iters))
    print('training the HolC model...')
    step = 0
    
    graph = tf.Graph()
    with graph.as_default():
        model = holcModel(_nsteps=_oplen ,_vocab_len=  embeddings.shape[0], _maxlen=_seqlen )
        sess = tf.Session(config=_config)
        with sess.as_default():
            # build holc model
            model.build()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.Variable(0,trainable=False)
                decayed_lr = tf.train.exponential_decay(config.lr, global_step, training_iters,decay_rate=0.95,staircase=False)
                if config.lr_method =='rmsprop': # rms optimizer
                    _optimizer = tf.train.RMSPropOptimizer(decayed_lr, epsilon=1e-6)
                elif config.lr_method =='adam': # adam optimizer
                    _optimizer = tf.train.AdamOptimizer(decayed_lr, epsilon=1e-6)
                elif config.lr_method =='sgd': # sgd optimizer
                    _optimizer = tf.train.GradientDescentOptimizer(decayed_lr/config.batch_size)
                elif config.lr_method =='adagrad': # adagrad optimizer
                    _optimizer = tf.train.AdagradOptimizer(decayed_lr)
                else:
                    print('Setup the optimization method!')
 
            with tf.name_scope('apply_gradient_norm'):
                gvs = _optimizer.compute_gradients(model.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                optimizer = _optimizer.apply_gradients(capped_gvs)
            
            # run and train the model
            sess.run(tf.global_variables_initializer())
     
            if config.pre_trained_embs:
                sess.run(model.embedding_init,feed_dict={model.embedding_placeholder: embeddings})
               
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/train', graph=tf.get_default_graph())
            dev_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/dev',graph=tf.get_default_graph())
            
            # initiate train/dev accuracies
            acc         = 0
            _acc        = 0
            _count      = 0
            _dev_loss   = 1.75 # arbitrary value, set lower than _train_loss
            _train_loss = 10  
            bias_variance_ratio = 5.00 # defines the diff in loss between train, dev 
            _kfold+=1
            print('kfold training #',_kfold)
            # keep training until reach max iterations
            while step <= training_iters:
                # get train batch
                batch_x, batch_y, batch_lengths,batch_oplengths =  data_helper.next_batch(config.batch_size, x_train,y_train, seqlen_train, True,op_lengths_train)
                
                # monitor training accuracy informati on
                summary,_ = sess.run([merged,optimizer], feed_dict={model.input_x: batch_x, model.input_y: batch_y,model.seqlen: batch_lengths, model.dropout: dyn_dropout, model.oplens:batch_oplengths, model.is_training:True})
                
                if _dev_loss < bias_variance_ratio * _train_loss: # optimize on train dataset
                    # get train batch
                    batch_x, batch_y, batch_lengths, batch_oplengths =  data_helper.next_batch(config.batch_size, x_train,y_train, seqlen_train, True,op_lengths_train)
                    # run optimization (backprop) on train dataset 
                    sess.run(optimizer, feed_dict={model.input_x: batch_x, model.input_y: batch_y, model.seqlen: batch_lengths,  model.dropout: dyn_dropout, model.oplens:batch_oplengths,model.is_training:True})
                    # calculate accuracy
                    acc = sess.run(model.accuracy, feed_dict={model.input_x: batch_x, model.input_y: batch_y ,model.seqlen: batch_lengths,  model.dropout: dyn_dropout, model.oplens:batch_oplengths,model.is_training:True})
                    # calculate loss
                    _train_loss = sess.run(model.loss, feed_dict={model.input_x: batch_x, model.input_y: batch_y ,model.seqlen: batch_lengths, model.dropout: dyn_dropout, model.oplens:batch_oplengths,model.is_training:True})
                    
                    if step % config.display_step == 0:
                        # output the results
                        print("Iter " + str(step) + ", Minibatch train Loss= " + \
                            "{:.6f}".format(_train_loss) + ", Accuracy= " + \
                            "{:.5f}".format(acc) + ", Dropout= " +\
                            "{:.2f}".format(dyn_dropout) +", Overfit:" +\
                            "{:.2f}".format((_count/config.overfit_threshold) *100)  + "%") 
                    
                if  bias_variance_ratio *_train_loss < _dev_loss: # optimize on dev dataset
                    # get dev batch
                    batch_x, batch_y, batch_lengths, batch_oplengths  =  data_helper.next_batch((config.batch_size, int(x_dev.shape[0]))[config.batch_size > int(x_dev.shape[0])], x_dev, y_dev, seqlen_dev, True,op_lengths_dev)
                    # run optimization (backprop) on dev sample
                    sess.run(optimizer, feed_dict={model.input_x: batch_x, model.input_y: batch_y, model.seqlen: batch_lengths,  model.dropout: dyn_dropout, model.oplens:batch_oplengths,model.is_training:True})
                    # calculate accuracy
                    acc = sess.run(model.accuracy, feed_dict={model.input_x: batch_x, model.input_y: batch_y ,model.seqlen: batch_lengths,  model.dropout: dyn_dropout, model.oplens:batch_oplengths,model.is_training:True})
                    # calculatge loss
                    _dev_loss = sess.run(model.loss, feed_dict={model.input_x: batch_x, model.input_y: batch_y ,model.seqlen: batch_lengths, model.dropout: dyn_dropout, model.oplens:batch_oplengths,model.is_training:True})
                    
                    if step % config.display_step == 0:
                        # output the results
                        print("Iter " + str(step) + ", Minibatch dev Loss= " + \
                            "{:.6f}".format(_dev_loss) + ", Accuracy= " + \
                            "{:.5f}".format(acc) + ", Dropout= " +\
                            "{:.2f}".format(dyn_dropout) +", Overfit:" +\
                            "{:.2f}".format((_count/config.overfit_threshold) *100)  + "%") 
                
                # monitor train accuracy information in python window
                if step % config.display_step == 0:
                    # get train batch
                    batch_x, batch_y, batch_lengths, batch_oplengths =  data_helper.next_batch(config.batch_size, x_train,y_train, seqlen_train, True,op_lengths_train)
                    # calculate train accuracy and add to summary
                    summary, acc = sess.run([merged,model.accuracy], feed_dict={model.input_x: batch_x, model.input_y: batch_y,model.seqlen: batch_lengths,  model.dropout: dyn_dropout,model.oplens:batch_oplengths,model.is_training:False})
                    
                    # add to summaries
                    train_writer.add_summary(summary, step)
                    
                # monitor dev accuracy information
                if step % config.display_step//2 == 0:
                    # get dev batch
                    batch_x, batch_y, batch_lengths,batch_oplengths  =  data_helper.next_batch((config.batch_size, int(x_dev.shape[0]))[config.batch_size > int(x_dev.shape[0])], x_dev, y_dev, seqlen_dev, True,op_lengths_dev)
                                        
                    summary, _acc = sess.run([merged,model.accuracy], feed_dict={model.input_x: batch_x, model.input_y: batch_y,model.seqlen: batch_lengths,  model.dropout: dyn_dropout,model.oplens:batch_oplengths, model.is_training:False})
                    
                    # add to summaries
                    dev_writer.add_summary(summary, step )
                    
                # calculate new dropout
                dyn_dropout, _count = calcMetric.calcDropout(acc,_acc,config.dropout,_count)
        
                # set a great value to prevent overfit early-stop
                if _count > config.overfit_threshold: 
                    print('Overfit Identidied')
                    step = training_iters
                    
                # increment training step
                step += 1
                
            print("Optimization Finished!")
            # evaluate test dataset
            test_len = int(x_test.shape[0])
            # for partial lists metrices
            list_partial_acc = []
            list_partial_cm = []
            partial_eval = False
            
            if test_len > config.test_eval_batch: 
                # mark the index of the data
                eval_idx_start = 0
                eval_idx_end = config.test_eval_batch - 1
                # mark partial evaluation
                partial_eval = True
                # partition the test data
                eval_range = (test_len//config.test_eval_batch) + 1
                for _ in range(eval_range):
                    tmp_test_data = x_test[eval_idx_start:eval_idx_end]
                    tmp_test_label = y_test[eval_idx_start:eval_idx_end]
                    tmp_test_seqs = seqlen_test[eval_idx_start:eval_idx_end]
                    tmp_op_lengths_test =op_lengths_test[eval_idx_start:eval_idx_end]
                    partial_acc = sess.run(model.accuracy, feed_dict={model.input_x: tmp_test_data,model.input_y:tmp_test_label, model.seqlen: tmp_test_seqs, model.dropout: dyn_dropout,model.oplens:tmp_op_lengths_test, model.is_training:False})
                    
                    # store partial accuracy
                    list_partial_acc.append(partial_acc)
                    
                    print("Partial Testing Accuracy:", partial_acc)
                    tmp_actual = np.array([np.where(r==1)[0][0] for r in tmp_test_label])
                    tmp_predicted = model.logits.eval(feed_dict={model.input_x: tmp_test_data, model.dropout: dyn_dropout,model.seqlen: tmp_test_seqs,model.oplens:tmp_op_lengths_test,model.is_training:False})
                      
                    cm = tf.confusion_matrix(tmp_actual,tmp_predicted,num_classes=config.n_classes)
                    # get confusion matrix values / store partial confusion matrix
                    list_partial_cm.append(sess.run(cm))
                    
                    # feed test evaluation with new values
                    eval_idx_start+=config.test_eval_batch
                    eval_idx_end +=config.test_eval_batch
                    if eval_idx_end > test_len:
                        eval_idx_end = test_len
                 
            else :
                # calculate overall accuracy
                accuracy = sess.run(model.accuracy, feed_dict={model.input_x: x_test,model.input_y:y_test, model.seqlen: seqlen_test, model.dropout: dyn_dropout,model.oplens:op_lengths_test,model.is_training:False})
                  
                # get actual labels
                actual = np.array([np.where(r==1)[0][0] for r in y_test])
                predicted = model.logits.eval(feed_dict={model.input_x: x_test, model.dropout: dyn_dropout,model.seqlen: seqlen_test,model.oplens:op_lengths_test,model.is_training:False})
                cm = tf.confusion_matrix(actual,predicted,num_classes=config.n_classes)
                # get confusion matrix values
                tf_cm = sess.run(cm)
                print(tf_cm)
                
            if partial_eval:
                 accuracy = np.ma.average(list_partial_acc) 
                 tf_cm = np.ma.sum(list_partial_cm,axis=0)
            
            print("\nOverall Testing Accuracy: ", accuracy)    
            
            print('Confusion Matrix: (H:labels, V:Predictions)')
            print('Precision | Recall | Fscore')
            if(y_train.shape[1]==2):
                print(calcMetric.pre_rec_fs2(tf_cm))
            elif (y_train.shape[1]==3):
                print(calcMetric.pre_rec_fs3(tf_cm))
            elif (y_train.shape[1]==4):
                print(calcMetric.pre_rec_fs4(tf_cm))
            elif (y_train.shape[1]==5):
                print(calcMetric.pre_rec_fs5(tf_cm))
            elif (y_train.shape[1]==6):
                print(calcMetric.pre_rec_fs6(tf_cm))
        
            metric_list.append(accuracy)
    
    print("")
    print("reseting default graph")
    tf.reset_default_graph()
    sess.close()
    
# tensorboard --logdir=/tmp/tensorflowlogs  
print('average acurracy: ' + "{:.2f}".format(np.average(metric_list)*100))
print(metric_list)
