#!/home/mathias/.venv/bin/python3.6
# -*- coding: utf-8 -*-
# Source: https://github.com/tianyic/LSTM-GRU/blob/master/src/lm_GRU
import sys
import numpy as np
import csv
import itertools
import nltk
import codecs
from collections import defaultdict
from datetime import datetime

"""
The frame of following code (includes class attribute, preprocessing function, and code organization) 
is adpted by Denny Britz, 
RECURRENT NEURAL NETWORKS TUTORIAL, 
posted on www.wilml.com: September 30, 215. 
The forward-prediction and back-propagation training is wirtten by
"""

def softmax( x, tau = 1.0 ):
    e = np.exp( np.array(x) / tau )
    return e / np.sum( e )

def sigmoid( x ):
    return 1 / ( 1 + np.e ** ( -x ) )

class GRUNumpy:

    def __init__(self, word_dim, hidden_dim = 30, bptt_truncate = 2 ):
        #Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Random asisgn weight as we did in RNNNumpy
        self.U = np.random.uniform( -np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim) )
        self.W = np.random.uniform( -np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, hidden_dim) )
        self.V = np.random.uniform( -np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim) )
        self.b = np.zeros( (3, hidden_dim) )
        self.c = np.zeros( word_dim )
        pass

    def forward_propagation( self, x):
        # The total number of time steps
        T = len(x)
        z = np.zeros( (T + 1, self.hidden_dim) )
        r = np.zeros( (T + 1, self.hidden_dim) )
        h = np.zeros( (T + 1, self.hidden_dim) )
        s = np.zeros( (T + 1, self.hidden_dim) )

        # s[-1] = np.zeros(self.hidden_dim)
        #print s.shape
        #print 'x:       '+str(x) 
        o= np.zeros( (T, self.word_dim))

        for t in np.arange(T):
            print("x[t]: {}".format(x[t]))
            z[t]=sigmoid( self.U[0,:,x[t] ] + self.W[0].dot(s[t-1]) + self.b[2] )
            r[t]=sigmoid( self.U[1,:,x[t] ] + self.W[1].dot(s[t-1]) + self.b[1] )
            h[t]=np.tanh( self.U[2,:,x[t] ] + self.W[2].dot( s[t-1]*r[t] ) + self.b[0] )
            s[t]=(1-z[t])*h[t]+z[t]*s[t-1]
            o[t]=softmax( self.V.dot(h[t]) + self.c)
        return [z, r, h, s, o]

    def predict( self, x): 
        z, r, h, s, o= self.forward_propagation( x )
        return np.argmax(o , axis = 1)

    def calculateLoss( self, x, y):
        L=0.0   
        # For each sentences
        N = np.sum( ( len(y_i) for y_i in y) )
        for i in np.arange( len(y) ):
            z, r, h, s, o = self.forward_propagation( x[i] )
            correct_word_predictions = o[ np.arange(len(y[i])), y[i] ]
            L += -1* np.sum( np.log( correct_word_predictions) )
        return L

    # See: https://github.com/tianyic/LSTM-GRU/blob/master/MTwrtieup.pdf
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        z, r, h, s, o = self.forward_propagation(x)

        # Then we need to calculate the gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape)
        dLdc = np.zeros(self.c.shape)

        delta_o = o
        delta_o[ np.arange(len(y)), y ] -= 1.0

        for t in np.arange(T)[::-1]:    # Reverse the order : [n ... 0]
            dLdV += np.outer( delta_o[t], s[t].T )
            delta_t = self.V.T.dot( delta_o[t] ) * ( 1 - ( s[t] ** 2 ) )
            dLdc += delta_o[t]
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:

                dLdW[0] += np.outer(delta_t, s[bptt_step-1])
                dLdU[0,:,x[bptt_step]] += delta_t
                dLdb[0] += delta_t

                dLdr = self.W[0].T.dot(delta_t) * (s[bptt_step-1])
                dLdW[1] += np.outer( dLdr*r[bptt_step]*(1-r[bptt_step]), s[bptt_step-1] )
                dLdU[1,:,x[bptt_step]] += dLdr*r[bptt_step]*(1-r[bptt_step])
                dLdb[1] += dLdr * r[bptt_step] * (1-r[bptt_step])

                if bptt_step>=1:
                    dLdz = self.W[0].T.dot(delta_t) * r[bptt_step] * (s[bptt_step-2]-h[bptt_step])
                    dLdW[2] += np.outer( dLdz * z[bptt_step] * (1-z[bptt_step]), s[bptt_step-1] )
                    dLdU[2,:,x[bptt_step] ] += dLdz * z[bptt_step] * (1-z[bptt_step])
                    dLdb[2] += dLdz * z[bptt_step] * (1-z[bptt_step])

        return [ dLdU, dLdV, dLdW, dLdb, dLdc ]

    def numpy_sdg_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW, dLdb, dLdc = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        self.b -= learning_rate * dLdb
        self.c -= learning_rate * dLdc
        
    def train_with_sgd(self,X_train, y_train, learning_rate=0.003, nepoch=200, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in np.arange( nepoch ):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculateLoss( X_train, y_train )
                losses.append( ( num_examples_seen, loss ) )
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f\n" % (time, num_examples_seen, epoch, loss))
                sys.stdout.flush()
            # For each training example...
            for i in np.arange(len(y_train)):
                self.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
    
    
    
def preprocessing( file, vocabulary_size, sentence_start_token, sentence_end_token, unknown_token ):
    sys.stderr.write( "Reading training file.\n" )

    with open( file, 'rb' ) as f:
        for x in f:
#            print(type(x[0]))
            break
        sentences = itertools.chain(*[nltk.sent_tokenize(chr(x[0]).lower()) for x in f])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
#        print(sentences)

    # tokenize sentences
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    #print X_train.shape
    return X_train, y_train

v_size = 2000
unknown_token = 'UNKNOWN_TOKEN'
start_token = 'SENTENCE_START'
end_token = 'SENTENCE_END'
filename = 'dataset/europarl_eng_10000.txt'
X_train, y_train = preprocessing( filename, v_size, start_token, end_token, unknown_token )