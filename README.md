# Description

This is my trial project after taking several Andrew Ng's deep learning courses, particularly the last one of 5 course series: "sequence model". I do not use python in daily work, by no means an experienced coder in Python.

It performs Named Entity Recognition (NER) using recurrent neural network (Bidirectional LSTM N to N network). Inspired by Jason Chiu & Eric Nichols's paper (only implemented 1/3 of what they do).

# Dependencies

* Anaconda Python 3.6.4 
* keras/tensorflow python packages, and other miscellaneous packages
* glove.6B.50d.bin: GloVe word vectors from Stanford, https://nlp.stanford.edu/projects/glove/

# Directory
## src
* bi_lstm_ner.py: main module for performing the training
* utils.py: utility functions
* preprocess_data.py: for data pre-processing

## data
* 3_tags_data.xml: sample data in XML tagging
* glove.6B.50d.txt: word vector data from Stanford. This file is too big for github, after downloading it, remember adding a line "400000 50" at top.


# Run

python Bi_LSTM_NER.py

Developed using Visual Studio Code with Python extension.


# Version Info
More documentation to come.

# Reference:
* Coursera course by Andrew Ng, Sequence Models: https://www.coursera.org/learn/nlp-sequence-models/home/welcome
* https://www.aclweb.org/anthology/Q16-1026

# Lessons and Future Work
* Spent majority coding effort on preparing the training data, and loading the word vector to set up the TensorFlow Embedding layer.
* Need to monitor the evaluation result during training.