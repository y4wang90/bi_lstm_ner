# The docker image is build on Windows 10. Running under windows container
#  To build docker image: docker build -t bi_lstm_ner .
#  To run docker image on windows 10: docker run --rm -it --memory=1g bi_lstm_ner cmd
#    Without "--memory=4g" options, the contianer gives memory error
#    "OOM when allocating tensor with shape[400002,50]"

FROM python:3.6

RUN     pip install numpy
RUN     pip install sklearn
RUN     pip install gensim
RUN     pip install keras

# Using an older version of tensorflow. The windows 10 box has NO GPU. 
# "pip install tensorflow" will install latest tensorflow version, 
# though specified as "cpu support only", it's still trying to load 
# GPU support package "CUDA"
RUN     pip install tensorflow==1.7
ADD lib/msvcp140.dll    ../Python/

# ADD C:\cygwin\Anaconda3 Anaconda3

ADD src/*.py            bi_lstm_ner/src/
ADD data                bi_lstm_ner/data/
ADD README.md           bi_lstm_ner/
ADD Dockerfile          bi_lstm_ner/

WORKDIR                 bi_lstm_ner/src
# CMD [ "python",      "bi_lstm_ner.py" ]