20180419
Training on NER using word2vec, LSTM

Environment:
Install Anaconda Python Windows 64 bits, version = 3.6.4
Install package 
    numpy:      http://www.numpy.org
    keras:      https://github.com/keras-team/keras
    tenserflow: https://www.tensorflow.org
    gensim:     http://radimrehurek.com/gensim python wrapper for word2vec

Load data 
    glove.6B.50d.txt: https://nlp.stanford.edu/projects/glove/
    Insert line "400000 50" at top
    
Files:
    ner_train.py:   loading word2vec model, setup network
    poi_300_train.xml: 300 samples of POI training data in XML format
    
Reference
    https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
    