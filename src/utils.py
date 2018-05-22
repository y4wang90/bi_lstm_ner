import numpy as np
import os.path
from gensim.models import KeyedVectors


"""
Convert one dimenional array to one hot matrix
params: X: shape = (m,)
        C: scaler, one hot vector dimension
return:
        OneHot: shape (m, dim)
"""
def convert_to_one_hot(A, c):
    out = np.zeros( (A.size,c), dtype=np.uint8)
    out[np.arange(A.size),A.ravel()] = 1
    out.shape = A.shape + (c,)
    return out

# def convert_to_one_hot(Y, C):
#     m = Y.shape[0]
#     OneHot = np.zeros((m, C))
#     for i in range(m):
#         OneHot[i,Y[i]] = 1.
#     return OneHot

def softmax(w):
    w = w - np.max(w)
    e = np.exp(w)
    dist = e / np.sum(e)
    return dist

"""
load the word vector model file
params:
    model_file: model file loaded from word2vec "glove.6B.50d.txt"
returns:
    wv: word vector
    iw: index vector
usage: typical model file name 'glove.6B.50d.txt'
"""
def load_wordvec_model(model_file):
    #Each corpus need to start with a line containing the vocab size and the vector size
    #in that order. So in this case you need to add this line "400000 50" as the first
    #line of the model.


    filename, file_extension = os.path.splitext(model_file)
    bin_file = filename + '.bin'
    if os.path.isfile(bin_file):
        model_file = bin_file
        print('loading model file \'{}\' ...'.format(model_file))
        wv = KeyedVectors.load_word2vec_format(bin_file, binary=True)
    else:
        print('loading model file \'{}\' ...'.format(model_file))
        wv = KeyedVectors.load_word2vec_format(model_file, binary=False)
        wv.save_word2vec_format(bin_file, binary=True)

    iw = wv.index2word

    wi = {v:k for k, v in enumerate(iw)}

    print('loaded model file \'{}\', word vector size {} x {}'.
        format(model_file, len(wv.index2word), wv.vector_size))
    return wv, iw, wi
