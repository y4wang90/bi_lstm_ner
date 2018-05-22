import numpy as np
import os.path
from datetime import datetime, date
import pickle

from keras.models import Model
from keras.layers import Bidirectional, LSTM
from keras.layers import Dense, Input, Dropout, Reshape
from keras.layers.embeddings import Embedding

from sklearn import metrics
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from utils import *
import preprocess_data as pd

def get_path(filename):
    return '../data/' + filename

"""
load the word vector model file
params:
    word_model_file: model file loaded from word2vec "glove.6B.50d.txt"
returns:
    wv: word vector
    iw: index vector
"""
def load_word_model(word_model_file, limit=None):
    #Each corpus need to start with a line containing the vocab size and the vector size
    #in that order. So in this case you need to add this line "400000 50" as the first
    #line of the model.
    global UNK, V_dim, Nw

    word_model_file_bin = '.'.join(word_model_file.split('.')[:-1]) + '.bin'

    if os.path.isfile(word_model_file_bin):
        print('loading model file \'{}\', limit={} ...'.format(word_model_file_bin, limit))
        wv = KeyedVectors.load_word2vec_format(word_model_file_bin, binary=True, limit=limit)
    else:
        print('loading model file \'{}\' ...'.format(word_model_file))
        wv = KeyedVectors.load_word2vec_format(word_model_file, binary=False, limit=limit)
        wv.save_word2vec_format(word_model_file_bin, binary=True)

    iw = {k: v for k, v in enumerate(wv.index2word, 1)}
    wi = {v: k for k, v in iw.items()}

    Nw = len(iw)
    Tx = 10
    UNK = Nw + 1
    V_dim = wv['the'].shape[0]

    print('loaded model file \'{}\', word vector size {} x {}'.
        format(word_model_file, len(wv.index2word), wv.vector_size))
    return wv, wi, iw


#hyper parameters
n_a                 = 32    # LSTM dimension, BiDirectional output dimension = 2 X n_a
sigmoid_threshold   = 0.5   # output sigmoid activation threshold

#global data
Tx = 20         # time steps, maximum number of words per training phrase
Nw = 0          # number of words
V_dim = 0       # vector dimensions
n_tags = 0      # number of NER tags
PAD = 0         # using mask_zero in Keras embedding layer, treat 0 as mask
UNK = 10        # unknown word index, make it maximum = vocab_size + 1
Unk_Words = []  # words not exist in word2index vector
nerTagsMap = {}

"""
preprocess the training data
params:
    training_phrases: training phrases as list
returns:
    train_data: np.array of shape (m, Tx), each elemented is a word index
"""
def preproc_data(training_phrases, training_tags, word2idx):
    global Unk_Words, n_tags

    n_phrase = len(training_phrases)
    X = np.zeros((n_phrase, Tx), dtype=np.int32)
    Y = np.zeros((n_phrase, Tx, n_tags), dtype=np.int32)
    for i in range(len(training_phrases)):
        n_tokens = min(len(training_tags[i]),Tx)
        y = np.array(training_tags[i][:n_tokens])
        Y[i,:n_tokens,:] = convert_to_one_hot(y, n_tags)

        for j in range(n_tokens):
            # ToDo: optionally remove possessive with a new feaure
            word = training_phrases[i][j].lower()
            #(UNK if word not in word2idx else word2idx[word])
            if (word in word2idx):
                X[i][j] = word2idx[word]
            else:
                X[i][j] = UNK
                if (word not in Unk_Words):
                    Unk_Words.append(word)
    return X, Y

def load_fake_model():
    wv = {  'the' :   np.array([0.22, 0.53, 0.43, 0.27]),       \
            'yellow': np.array([0.1,  0.5,  0.3,  0.7]),        \
            'four':   np.array([0.2,  1.2,  0.4,  0.9])         \
    }

    iw = {  1: 'the',                   \
            2: 'yellow',                \
            3: 'four'                   \
    }

    wi = {v: k for k, v in iw.items()}

    return wv, wi, iw


def createEmbeddingLayer(word2vce, word2idx):
    # 0 for mask_zero, Nw+1 for UNK word
    embedding_weights = np.zeros((Nw+2, V_dim))
    for word, index in word2idx.items():
        embedding_weights[index, :] = word2vec[word]

    embedding_layer = Embedding(input_dim=Nw+2, output_dim=V_dim, mask_zero=True, trainable=False)
    return embedding_layer

def createBaseModel(word2vec, word2idx):
    global n_tags

    # 0 for mask_zero, Nw+1 for UNK word
    embedding_weights = np.zeros((Nw+2, V_dim))
    for word,index in word2idx.items():
        embedding_weights[index, :] = word2vec[word]

    # create the model
    input = Input(shape=(Tx,), name='input', dtype=np.int32)

    embedding_layer = Embedding(input_dim=Nw+2, output_dim=V_dim, mask_zero=True, trainable=False)
    embedding_layer.build((None,)) # if you don't do this, the next step won't work
    embedding_layer.set_weights([embedding_weights])

    bi_lstm_layer = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(Tx, 2*n_a), name='bi-lstm')

    emb_vec = embedding_layer(input)
    a = bi_lstm_layer(emb_vec)
    if use_dropout:
        a = Dropout(rate=0.5, name="post_lstm_drop")(a)
    output = Dense(n_tags, activation='softmax', name='post-lstm-dense-softmax')(a)

    model = Model(input=input, output=output)
    #model.summary()
    return model


def displayTrainData(phrase, tags):
    for i in range(len(phrase)):
        print("{} : {}".format(' '.join(phrase[i]), tags[i]))

def displayTestOutput(phrases_raw, phrase, tags, out_tags):
    n_err = 0
    for i in range(len(phrase)):
        out_list = list(out_tags[i][:len(tags[i])])
        if (tags[i] != out_list):
            oov = []
            for j in range(len(phrase[i])):
                if phrase[i][j] in Unk_Words: oov.append(phrase[i][j])
            if len(oov) > 0: print('\n{}   oov={}'.format(phrases_raw[i].strip(), oov))
            else           : print('\n{}'.format(phrases_raw[i].strip()))
            print("{}".format(tags[i]))
            print("{}".format(out_list))
            n_err = n_err + 1

    return n_err

def evalOutput(tags, out_tags):
    # no time to search "converting o"
    expected_tags_array = np.zeros(out_tags.shape, dtype=np.int32)
    for i in range(len(tags)):
        expected_tags_array[i][:len(tags[i])] = np.array(tags[i])
        out_tags[i][len(tags[i]):] = 0

    conf_matrix = metrics.confusion_matrix(expected_tags_array.flatten(), out_tags.flatten())
    f1_scores = metrics.f1_score(expected_tags_array.flatten(), out_tags.flatten(), average=None)
    f1_scores = np.round(f1_scores, 3)
    print(f1_scores)
    print(nerTagsMap)
    print(conf_matrix)

    return f1_scores

def verify_and_split_train_test(data_file):
    name, extension = os.path.splitext(data_file)
    train_file = name + '_train' + extension
    test_file = name + '_test' + extension
    base_name = os.path.basename(name)
    if (not os.path.isfile(train_file)) or (not os.path.isfile(test_file)):
        with open(data_file) as f:
            content = f.readlines()
            train_phrases, test_phrases = train_test_split(content, test_size=0.1)

        with open(train_file,'w') as f:
            f.write( ''.join( train_phrases ) )

        with open(test_file,'w') as f:
            f.write( ''.join( test_phrases ) )

    return base_name, train_file, test_file

xml_data_file   = '3_tags_data.xml'
word_model_file = 'glove.6B.50d.txt'
Nw_to_load      = 400000
use_dropout     = True

def main():
    global Tx, Nw, n_tags, nerTagsMap, Unk_Words

    """
    generate training data
    training_phrases: training phrase as a list, each list entry is a list of tokens
    training_tags:    tag for each phrase, each entry is a list of tagging for phrase
    """

    data_file_name, train_data_file, test_data_file = verify_and_split_train_test(get_path(xml_data_file))

    # for multiple tags, the the training tags shape = (m, Tx, n_tags), matrix of one-hot vector
    # n_tags is number of tags plus 1 (for 0 label)
    train_phrases, train_tags, nerTagsMap, train_phrases_raw = pd.read_xml_train_data(train_data_file, nerTagsMap)
    Tx = len(max(train_phrases, key=len))
    Tx = min(20, max(Tx, 10))   # time step is between 10 and 20

    n_tags = (len(nerTagsMap) + 1)
    print("loaded train data {}: shape=({},{}), ner tags={}".format(train_data_file, len(train_phrases), Tx, nerTagsMap))

    wv, wi, iw  = load_word_model(get_path(word_model_file), Nw_to_load)
    Nw = len(iw)
    assert(Nw == len(wv.vocab))

    # create model
    model = createBaseModel(wv, wi)

    # file nameing convention
    #       data_file         = '3_tags_data.xml'
    #       word_model_file   = 'glove.6B.50d.txt'
    # If loading first 10000 word vector from 'glove.6B.50d.txt', it will have 2 file
    #       binary model file   = '3_tags_data_glove.6B.50d_w10000.h5'
    #       auxiliary conf gile = '3_tags_data_glove.6B.50d_w10000.aux'
    word_model_file_base = '.'.join(os.path.basename(word_model_file).split('.')[:-1])
    modelName = data_file_name + '_' + word_model_file_base + '_w' + str(Nw)
    modelFile = get_path(modelName + '.h5')
    modelFileAux = get_path(modelName + '.aux')
    if os.path.isfile(modelFile):
        model.load_weights(modelFile)
        nerTagsMap = pickle.load( open(modelFileAux, "rb") )

    else:
        print("train_phrase size {}, train_tags size {} ".format(len(train_phrases), len(train_tags)))
        X, Y = preproc_data(train_phrases, train_tags, wi)

        loss_name = 'categorical_crossentropy'
        model.compile(optimizer='Adam', loss=loss_name, metrics=['accuracy'])

        model.fit(X, Y, epochs=100, verbose=2)

        model.save(modelFile)
        pickle.dump( nerTagsMap, open(modelFileAux, "wb" ) )

    test_phrases, test_tags, _, test_phrases_raw = pd.read_xml_train_data(test_data_file, nerTagsMap)
    assert(len(test_phrases_raw) == len(test_phrases))

    X_test, Y_test = preproc_data(test_phrases, test_tags, wi)
    Y_out = model.predict(X_test)
    Y_out = np.argmax(Y_out, axis=-1)   # get index from softmax output

    print('\nTrain data len={}, Tx={}, wordvector ({} X {})'.format(len(train_phrases), Tx, Nw, V_dim))
    #displayTrainData(train_phrases[:5], train_tags[:5])

    n_err = displayTestOutput(test_phrases_raw, test_phrases, test_tags, Y_out)

    print('\nUnknown word={}'.format(len(Unk_Words)))
    print('\nY_out shape=' + str(Y_out.shape))
    F1 = evalOutput(test_tags, Y_out)

    with open(get_path('ner_result.txt'), 'a') as f:
        strModel = 'embedding->bi_lstm' + ('->drop' if use_dropout else '') + '->dense(softmax)'
        f.write('{:%Y.%m.%d %H:%M}: model={}, data={}, wordvec={}\n'.format(datetime.now(), strModel, data_file_name, word_model_file))
        f.write('tags={}, F1={}\n'.format(nerTagsMap, F1[1:]))
        f.write("n_train=%d, n_test=%d, n_err=%d, n_tags=%s, n_unkWord=%d, Tx=%d, vocab=%d\n\n" \
            % (len(train_phrases), len(test_phrases), n_err, n_tags, len(Unk_Words), Tx, Nw))


if __name__ == "__main__":
    # execute only if run as a script
    main()