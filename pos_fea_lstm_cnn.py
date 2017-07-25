import os,sys
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu1"#,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"#,lib.cnmem=1"
import numpy as np
np.random.seed(1999) # for reproducibility

from keras.models import *
from keras.regularizers import l2
from keras.layers.core import *
from keras.layers import Input, merge, Embedding, LSTM, TimeDistributed
from keras.layers.convolutional import Convolution1D, MaxPooling1D,Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils
from gensim.models import Word2Vec
import re
from gensim.parsing import strip_multiple_whitespaces
from w2v import train_word2vec
from collections import Counter
import itertools
from layers import MaxPiecewisePooling1D
from log import log_error


#basic superparameter
w2c_len = 30
dropout_prob = (0.25,0.5)
num_filters = 150
filter_sizes = (3, 4)
hidden_dims = 150
hidden_dims_for_manual = 180
nb_epoch = 10
batch_size = 32
val_size = 0.1
pos_len = 5
sparse_fea = 0


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


chemical_label_sentence, disease_label_sentence = "entc", "entd"
chemical_label_sentence_end , disease_label_sentence_end =  "entcend", "entdend"


def build_pos_flag(sentence, sequence_length):

    def pos_flag(label_start, label_end):
        start = sentence.index(label_start)
        end   = sentence.index(label_end)
        pos_list = [sequence_length]*sequence_length
        for i in xrange(0,start):
            pos_list[i] = i - start
        for i in xrange(start,end):
            pos_list[i] = 0
        for i in xrange(end, len(sentence)):
            pos_list[i] = i - end
        return pos_list
    pos_1 = pos_flag(chemical_label_sentence,chemical_label_sentence_end)
    pos_2 = pos_flag(disease_label_sentence,disease_label_sentence_end)

    return pos_1, pos_2

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

def build_input_data(sentences, labels, vocabulary, pos1_sentences, pos2_sentences):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    a1 = np.array(pos1_sentences)
    a2 = np.array(pos2_sentences)
    return [x, y, a1, a2]



class input_data:
    def __init__(self,head,btwn,tail,y,vocabulary):
        self.vocabulary = vocabulary
        self.head = self.reads(head)    #head contains [head_text, pos1_index, pos2_index]
        self.btwn = self.reads(btwn)
        self.tail = self.reads(tail)
        self.y = self.read_y(y)

    def get_x(self):
        return [self.head[0],self.btwn[0],self.tail[0]]

    def get_index(self):
        head_index = self.head[0].shape[1]
        between_index = head_index + self.btwn[0].shape[1]
        tail_index = between_index + self.tail[0].shape[1]
        return [head_index,between_index,tail_index]

    def get_x_concatenate(self):
        #print self.head[0]
        return np.concatenate((self.head[0],self.btwn[0],self.tail[0]),axis=1)

    def get_pos_concatenate(self):
        return [np.concatenate((self.head[1],self.btwn[1],self.tail[1]),axis=1), np.concatenate((self.head[2],self.btwn[2],self.tail[2]),axis=1)]

    def reads(self,head):
        x = self.read_x(head)
        pos1,pos2 =self.read_pos(head)
        return [x,pos1,pos2]

    def read_x(self, head):
        return np.array([[self.vocabulary[word] for word in sentence] for sentence in head[0]])

    def read_pos(self,head):
        return np.array(head[1]),np.array(head[2])

    def read_y(self,y):
        return np.array(y)


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

#return train_x, train_y, test_x, test_y, sentence_length
def read_data(trainfile,testfile,w2c_file):
    #file to padded_sentences
    train_all_text, train_head, train_btwn, train_tail, train_y, max_lengths =  data2numpy(trainfile)
    test_all_text, test_head, test_btwn, test_tail, test_y, max_lengths = data2numpy(testfile,max_lengths=max_lengths,mode='test')

    #map to vocabulary
    vocabulary, vocabulary_inv = build_vocab(train_all_text + test_all_text )

    train_datas , test_datas = input_data(train_head, train_btwn, train_tail,train_y, vocabulary),\
                               input_data(test_head, test_btwn, test_tail, test_y, vocabulary)

    return train_datas, test_datas, max_lengths, vocabulary, vocabulary_inv

#with split one sentence to three part, head, between and tail.
def data2numpy(filename,max_lengths = [], mode = 'train'):
    dataset = open(filename).read().strip().split('\n')
    index = []
    x,y,datas = [],[],[]
    for i,data in enumerate(dataset):
        label, sentence = data.split('\t')
        if label.strip() == "1":
            y.append(1)
        else:
            y.append(0)
        datas.append(sentence)
        index.append(i)

    x_text = [clean_str(sentence) for sentence in datas]        #Tokenization
    x_text = [s.split(" ") for s in x_text]                     #split
    #split it to three part, each part contains [sentences, index_chemical, index_disease]
    head, between, tail, max_lengths = split_x_add_padding(x_text,max_lengths,mode)
    x_text = pad_sentences(x_text)
    return x_text, head, between, tail, y, max_lengths



def split_x_add_padding(x_text,max_lengths, mode):
    #position added
    a1, a2 = [], []
    head, between,tail = [],[],[]
    for sentence in x_text:
        a1_pos, a2_pos = build_pos_flag(sentence, len(sentence))
        a1.append(a1_pos)
        a2.append(a2_pos)

        c_start, c_end, d_start, d_end = sentence.index(chemical_label_sentence),sentence.index(chemical_label_sentence_end),\
                                     sentence.index(disease_label_sentence), sentence.index(disease_label_sentence_end)
        if c_start > d_start:            #if not chemical first, switch them
            c_start, d_start = d_start, c_start
            c_end, d_end = d_end, c_end

        #add sentence and pos info to head\b\t
        head.append([sentence[:c_end+1], a1_pos[:c_end+1], a2_pos[:c_end+1]])
        between.append([sentence[c_start:d_end+1], a1_pos[c_start:d_end+1], a2_pos[c_start:d_end+1]])
        tail.append([sentence[d_start:], a1_pos[d_start:], a2_pos[d_start:]])

    head ,between, tail = zip(*head),zip(*between),zip(*tail)

    if mode == 'train':
        head_max = max(len(sen) for sen in head[0])
        between_max = max(len(sen) for sen in between[0])
        tail_max = max(len(sen) for sen in tail[0])
        max_lengths = [head_max, between_max, tail_max]

    head = add_padding(sentences=head[0],a1=head[1],a2=head[2],sequence_length=max_lengths[0])
    between = add_padding(between[0],between[1],between[2], sequence_length = max_lengths[1])
    tail = add_padding(tail[0],tail[1],tail[2],sequence_length=max_lengths[2])

    return head, between, tail, max_lengths



def add_padding(sentences, a1, a2, padding_word="<PAD/>", sequence_length = 0):
    if sequence_length == 0:
        sequence_length = max(len(x) for x in sentences)
    padded_sentences, pos1_sentences, pos2_sentences = [],[],[]
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length -len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        a = a1[i] + [sequence_length] * num_padding
        b = a2[i] + [sequence_length] * num_padding
        padded_sentences.append(new_sentence)
        pos1_sentences.append(a)
        pos2_sentences.append(b)
    return padded_sentences, pos1_sentences, pos2_sentences



def get_embedding_weights(train_x,test_x,vocabulary_inv,min_count=1, context = 10):
    x = np.concatenate((train_x,test_x),axis=0)
    return train_word2vec(x, vocabulary_inv, w2c_len, min_count, context)


def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans


def get_Y(X, xmaxlen):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim


def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans


def add_weight(X):
    Y, alpha = X[0], X[1].dimshuffle(0,1,'x') #K.permute_dimensions(X[1],(1,2,0))
    return Y * alpha




def model_load(max_lengths,index, embedding_weights, vocabulary, manual_length):


    #################CNN0#######################
    sentence_input = Input(shape=(max_lengths,),dtype='int32',name='sentence_input')
    myembed = Embedding(len(vocabulary), w2c_len, input_length=max_lengths,
                        weights=embedding_weights)(sentence_input)

    pos_input1 = Input(shape=(max_lengths,),dtype='int32',name='pos_input1')
    p1embed = Embedding(max_lengths*2+1, pos_len,input_length=max_lengths)(pos_input1)

    pos_input2 = Input(shape=(max_lengths,),dtype='int32',name='pos_input2')
    p2embed = Embedding(max_lengths*2+1, pos_len,input_length=max_lengths)(pos_input2)

    m = merge([myembed,p1embed,p2embed],mode='concat',concat_axis=-1 )


    ##############attention 1#################
    '''
    lstm_units = 30
    lstm_fwd = LSTM(lstm_units, return_sequences=True)(m)
    lstm_bwd = LSTM(lstm_units, return_sequences=True, go_backwards=True)(m)
    bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
    drop_l1 = Dropout(dropout_prob[0]/2)(bilstm)
    h_n = Lambda(get_H_n, output_shape=(lstm_units*2,), name="h_n")(drop_l1)
    Y = Lambda(get_Y, arguments={"xmaxlen": max_lengths}, name="Y", output_shape=(max_lengths, lstm_units*2))(drop_l1)
    Whn = Dense(lstm_units*2 , W_regularizer=l2(0.01), name="Wh_n")(h_n)
    Whn_x_e = RepeatVector(max_lengths, name="Wh_n_x_e")(Whn)
    WY = TimeDistributed(Dense(lstm_units*2, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name="merged", mode='sum')
    M = Activation('tanh', name="M")(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(max_lengths, activation='softmax', name="alpha")(flat_alpha)

    r_ = merge([myembed,alpha],output_shape = (None,max_lengths,w2c_len), mode=add_weight)
    #r1 = Reshape((max_lengths,),name = 'r1')(r_)
    m = merge([r_,p1embed,p2embed],mode='concat',concat_axis=-1)
    '''

    ########## piecewise CNN ##########
    drop1 = Dropout(dropout_prob[0])(m)
    cnn2 = [Convolution1D(nb_filter=num_filters,
                     filter_length= fsz,
                     border_mode='valid',
                     activation='relu',
                     subsample_length=1)(drop1) for fsz in filter_sizes]


    # add attention
    from layers import ConvAttention
    attention = [ConvAttention(attention_dim =6)(item) for item in cnn2]

    pool2 = [MaxPiecewisePooling1D(pool_length=2, split_index=index)(item) for item in attention]

    flatten2 = [Flatten()(pool_node) for pool_node in pool2]
    merge_cnn2 = merge(flatten2,mode='concat')
    x2 = Dense(hidden_dims,activation='relu')(merge_cnn2)
    x3 = Dropout(dropout_prob[1])(x2)

    manual_input = Input(shape=(manual_length,),dtype='float32',name='manual_input')
    if sparse_fea == 0:
        x4 = Dense(hidden_dims_for_manual*2,activation='relu')(manual_input)
        x4 = Dense(hidden_dims_for_manual,activation='relu')(x4)
        #x4 = Dense(hidden_dims_for_manual,activation='relu')(x4)
        m3 = merge([x3,x4],mode='concat')
    else:
        m3 = merge([x3,manual_input],mode='concat')

    main_loss = Dense(1,activation='sigmoid',name='main_output')(m3)

    model =  Model(input= [sentence_input, pos_input1, pos_input2,
                            manual_input], output=main_loss)
    model.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])

    return model#, out_layer


def model_save(model, model_file):
    json_string = model.to_json()
    open( model_file+'.json', 'w').write(json_string)
    model.save_weights( model_file + '.h5',overwrite=True)


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
    p = right/(right+wrong) if right+ wrong != 0 else 0.1
    r = right/(right+miss)  if right + miss != 0 else 0.1
    f = 2*p*r/(p+r) if p+r != 0 else 0.0
    #print p,r,f
    return p,r,f



###model check for each epoch
class CheckBench(Callback):
    def __init__(self,test_data,test_y):
        self.test_data = test_data
        self.test_y = test_y
        self.max_fscore = 0.0
        self.max_info = {}
        self.counter = 0

    def on_batch_end(self,batch, logs={}):
        #for search faster
        #if batch < 200:
        #    return 1
        #result = self.model.predict(self.test_data,batch_size = batch_size)
        result = self.model.predict(self.model.validation_data[:4], batch_size=batch_size)
        #p,r,f = fscore(self.test_y,result)
        p,r,f = fscore(self.model.validation_data[-3],result)
        if f > self.max_fscore:
            self.max_fscore = f
            self.max_info['p'] = p
            self.max_info['r'] = r
            self.max_info['fscore'] = f
            self.max_info['batch'] = batch
            if f > 0.65:
                #model_save(self.model, "best_model_save")
                print "*************In test data**************"
                result_test =  self.model.predict(self.test_data,batch_size=batch_size)
                print "Best PRF:",fscore(self.test_y,result_test)
                np.savetxt("best_pcnn_feature_save.txt",result_test)
                print "***************************************"
            print "PRF on val-data:", p,r,f,batch

    def log_out(self,predict,golden,log_name):
        log_error(testfile,predict,golden,log_name)

    def on_epoch_end(self,epoch,logs={}):
        print "==================epoch end========================"
        #result = self.model.predict(self.test_data,batch_size=batch_size)
        #print fscore(self.test_y,result)
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

    #val_point = int((1-val_size)*len(train_x))
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
    #result_y = model.predict(test_x,batch_size =batch_size)
    #print result_y
    #np.savetxt(result_output,result_y)
    #model_save(model,model_output)
    return model


def feature_manual(feature_file, length = -1):
    from text_svm import svm_format_load as sfl
    feature_list = []
    for ff in feature_file:
        feature_list.append( sfl(ff,x_format='array')[0] ) #add fea to label_list
    #print feature_list
    tmp = feature_list[0]
    for fl in feature_list[1:]:
        tmp = np.append(tmp,fl,axis =1)
    print tmp.shape
    if length > 1:
        ap = np.zeros((tmp.shape[0],length - tmp.shape[1]))
        tmp = np.append(tmp,ap, axis = 1)
    return tmp


def processing(trainfile, testfile, train_feature_file, test_feature_file):
    #training && test
    train_datas, test_datas,\
    max_lengths, vocabulary, vocabulary_inv = read_data(trainfile= trainfile,
                              testfile = testfile, w2c_file= './data/newbin')
    embedding_weights = get_embedding_weights(train_datas.get_x_concatenate(), test_datas.get_x_concatenate(),vocabulary_inv)

    train_x = [train_datas.get_x_concatenate()] + train_datas.get_pos_concatenate()
    test_x  = [test_datas.get_x_concatenate()]  + test_datas.get_pos_concatenate()

    train_index = train_datas.get_index()
    test_index  = test_datas.get_index()

    train_y,test_y = train_datas.y , test_datas.y

    #add features
    manual_train = feature_manual(train_feature_file)
    manual_test  = feature_manual(test_feature_file, length = manual_train.shape[1])

    print manual_train.shape, manual_test.shape
    manual_length = manual_train.shape[1]

    train_x.append(manual_train)
    test_x.append(manual_test)

    print train_x

    model = model_load(max_lengths=sum(max_lengths),index = train_index, embedding_weights=embedding_weights, vocabulary=vocabulary, manual_length = manual_length)
    model_run(model,train_x,train_y,test_x, test_y,"./result_report/result_cnn.txt", "./data/cnn_model")
    #out_run(out_layer,train_x, train_y, test_x, test_y, "./data/cnn_train_data.svm", "./data/cnn_test_data.svm")

    #benchmark
    #from benchmark import benchmark_cnn
    #benchmark_cnn("./result_report/result_cnn.txt","./data/here_test")




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    trainfile, testfile = sys.argv[1], sys.argv[2]

    train_feature_file = ['./data/train_medi_fea.svm', './data/train_ctd_fea.svm', \
    './data/train_mesh_fea.svm', './data/train_sider_fea.svm',
    './data/train_mention_fea.svm']
    test_feature_file  = ['./data/test_medi_fea.svm', './data/test_ctd_fea.svm', \
    './data/test_mesh_fea.svm', './data/test_sider_fea.svm',
    './data/test_mention_fea.svm']

    processing(trainfile, testfile,train_feature_file, test_feature_file)

