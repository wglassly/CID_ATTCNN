import os,sys
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu1"
from keras.models import Sequential, Graph, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Input, merge, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D,Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils
from gensim.models import Word2Vec
import numpy as np
import re
from gensim.parsing import strip_multiple_whitespaces
from w2v import train_word2vec
from collections import Counter
import itertools

#basic superparameter
w2c_len = 30
dropout_prob = (0.25,0.5)
num_filters = 150
filter_sizes = (3, 4)
hidden_dims = 150
nb_epoch = 10
batch_size = 32
val_size = 0.1
np.random.seed(1337) # for reproducibility


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(sentences, padding_word="<PAD/>",sequence_length = 0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if sequence_length == 0:
        sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

#return train_x, train_y, test_x, test_y, sentence_length
def read_data(trainfile,testfile,w2c_file):
    #file to padded_sentences
    train_text , train_y, max_length = data2numpy(trainfile)
    test_text,  test_y  = data2numpy(testfile, max_length = max_length, mode = 'test')
    #map to vocabulary
    vocabulary, vocabulary_inv = build_vocab(train_text + test_text)

    train_x , train_y = build_input_data(train_text,train_y,vocabulary)
    test_x,   test_y  = build_input_data(test_text, test_y, vocabulary)

    return train_x,train_y,test_x,test_y, max_length, vocabulary, vocabulary_inv


def data2numpy(filename,max_length = 0, mode = 'train'):
    dataset = open(filename).read().strip().split('\n')

    x,y,datas = [],[],[]
    for data in dataset:
        label, sentence = data.split('\t')
        if label.strip() == "1":
            y.append(1)
        else:
            y.append(0)
        datas.append(sentence)

    x_text = [clean_str(sentence) for sentence in datas]
    x_text = [s.split(" ") for s in x_text]

    if mode == 'train':
        max_length = max(len(sen) for sen in x_text)

    x = pad_sentences(x_text, sequence_length=max_length)
    if mode == 'train':
        return x,y,max_length
    else:
        return x,y

def get_embedding_weights(train_x,test_x,vocabulary_inv,min_count=1, context = 10):
    x = np.concatenate((train_x,test_x),axis=0)
    return train_word2vec(x, vocabulary_inv, w2c_len, min_count, context)

def model_load(sequence_length, embedding_weights, vocabulary):

    sentence_input = Input(shape=(sequence_length,),dtype='int32',name='sentence_input')
    myembed = Embedding(len(vocabulary), w2c_len, input_length=sequence_length,
                        weights=embedding_weights)(sentence_input)
    drop1 = Dropout(dropout_prob[0])(myembed)
    cnn2 = [Convolution1D(nb_filter=num_filters,
                     filter_length= fsz,
                     border_mode='valid',
                     activation='relu',
                     subsample_length=1)(drop1) for fsz in filter_sizes]



    pool2 = [MaxPooling1D(pool_length=2)(item) for item in attention]
    flatten2 = [Flatten()(pool_node) for pool_node in pool2]
    merge_cnn2 = merge(flatten2,mode='concat')
    x2 = Dense(hidden_dims,activation='relu')(merge_cnn2)
    x3 = Dropout(dropout_prob[1])(x2)
    main_loss = Dense(1,activation='sigmoid',name='main_output')(x3)

    model =  Model(input= sentence_input, output=main_loss)
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    out_layer = Model(input = sentence_input, output = x2)
    out_layer.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model, out_layer


def model_save(model, model_file):
    json_string = model.to_json()
    open( model_file+'.json', 'w').write(json_string)
    model.save_weights( model_file + '.h5',overwrite=True)


###model check for each epoch
class CheckBench(Callback):
    def __init__(self,test_data,test_y):
        self.test_data = test_data
        self.test_y = test_y
        self.max_fscore = 0.0
        self.max_info = {}
        self.counter = 0

    def on_batch_end(self,batch, logs={}):

        result = self.model.predict(self.model.validation_data[:1], batch_size=batch_size)
        p,r,f = fscore(self.model.validation_data[-3],result)
        if f > self.max_fscore:
            self.max_fscore = f
            self.max_info['p'] = p
            self.max_info['r'] = r
            self.max_info['fscore'] = f
            self.max_info['batch'] = batch
            if f > 0.45:
                #model_save(self.model, "best_model_save")
                print "*************In test data**************"
                result_test =  self.model.predict(self.test_data,batch_size=batch_size)
                print "Best PRF:",fscore(self.test_y,result_test)
                np.savetxt("best_standard_cnn_model_save.txt",result_test)
                print "***************************************"
            print "PRF on val-data:", p,r,f,batch

    def log_out(self,predict,golden,log_name):
        log_error(testfile,predict,golden,log_name)

    def on_epoch_end(self,epoch,logs={}):
        print "==================epoch end========================"
        self.counter += 1



'''
Split dataset to train and dev.
input: ALL train dataset or train label
output: (train,dev)
'''
def split_x(train_x, val_size):
    if type(train_x) == type([]):
        val_point = int((1-val_size)*len(train_x[0]))
        return [data[:val_point] for data in train_x] , [data[val_point:] for data in train_x]
    else:
        val_point = int((1-val_size)*len(train_x))
        return train_x[:val_point], train_x[val_point:]

def model_run(model,train_x,train_y,test_x,test_y,\
                result_output,
                model_output,
                batch_size=batch_size,
                nb_epoch= nb_epoch,
                validation_split = val_size):

    '''
        run model with stable mode, without
    '''
    t_x, v_x = split_x(train_x,validation_split)
    t_y, v_y = split_x(train_y ,validation_split)


    save_epoch_result = CheckBench(test_data=test_x,test_y = test_y)                                        #save each epoch result
    model.fit(t_x,t_y,batch_size=batch_size,nb_epoch=nb_epoch,
                #validation_split=val_size,
                validation_data = (v_x,v_y),
                verbose=2,
                callbacks=[save_epoch_result])  # without split validation_data use test as val

    return model


def out_run(out_layer, train_x, train_y, test_x,test_y, output_train_layer, output_test_layer):
    result_train = out_layer.predict(train_x)
    result_test  = out_layer.predict(test_x)

    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(result_train,train_y,output_train_layer)
    dump_svmlight_file(result_test, test_y, output_test_layer )

    return 1

def fscore(y_test, y_predict):
    right , wrong, miss = 0.0, 0.0, 0.0
    #print y_test
    for i,j in zip(y_test, y_predict):
        #i = 1 if i[0]<i[1] else 0
        #j = 1 if j[0]<j[1] else 0
        #print i
        if type(i) == np.array([]):
            i = i[0]
        if i == 1 and j >= 0.5:
            right += 1
        elif i == 1 and j < 0.5:
            miss += 1
        elif i == 0 and j>= 0.5:
            wrong += 1
    p = right/(right+wrong) if right+ wrong != 0 else 0.001
    r = right/(right+miss)  if right + miss != 0 else 0.001
    f = 2*p*r/(p+r) if p+r != 0 else 0.0
    #print p,r,f
    return p,r,f


def processing(trainfile, testfile):
    #training && test
    train_x, train_y, test_x, test_y,\
    sentence_length, vocabulary, vocabulary_inv = read_data(trainfile= trainfile,
                                                            testfile = testfile, w2c_file= './data/newbin')
    print train_x
    embedding_weights = get_embedding_weights(train_x,test_x,vocabulary_inv)

    model, out_layer = model_load(sequence_length=sentence_length, embedding_weights=embedding_weights, vocabulary=vocabulary)
    model_run(model,train_x,train_y,test_x, test_y,"./result_report/result_cnn.txt", "./data/cnn_model")

    #benchmark
    #from benchmark import benchmark_cnn
    #benchmark_cnn("./result_report/result_cnn.txt","./data/here_test")




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    trainfile, testfile = sys.argv[1], sys.argv[2]
    processing(trainfile, testfile)
